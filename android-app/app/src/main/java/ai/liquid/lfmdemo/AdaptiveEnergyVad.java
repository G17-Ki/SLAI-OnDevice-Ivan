package ai.liquid.lfmdemo;

/**
 * A small, low-allocation voice activity detector for 16 kHz PCM16 capture.
 *
 * <p>The detector deliberately owns no audio. The recorder should keep a PCM ring buffer sized
 * from {@link Config#getPreRollSamples()} and use the sample offsets carried by callback events to
 * cut an utterance. Callbacks are synchronous on the caller's thread and must return quickly.</p>
 *
 * <p>This is an adaptive energy gate, not a neural VAD. It is intended as a deterministic first
 * Android implementation and should eventually be compared with a neural detector on real noisy
 * microphone recordings.</p>
 */
public final class AdaptiveEnergyVad {
    public static final int SAMPLE_RATE_HZ = 16_000;
    private static final double PCM16_SCALE = 32_768.0;
    private static final double MIN_DBFS = -100.0;

    public enum State {
        IDLE,
        LISTENING,
        SPEECH,
        HANGOVER,
        ENDPOINT
    }

    public enum EndpointReason {
        NONE,
        TRAILING_SILENCE,
        MAX_UTTERANCE
    }

    public enum EventType {
        STATE_CHANGED,
        SPEECH_CONFIRMED,
        ENDPOINT,
        SHORT_BURST_REJECTED,
        CANCELLED
    }

    /**
     * DISABLED ignores microphone energy while assistant audio is active. DETECT_ONLY reports
     * barge-in without prescribing playback cancellation. CANCEL_OUTPUT_ON_SPEECH sets the
     * event's cancel-output recommendation when speech is confirmed during assistant playback.
     */
    public enum BargeInPolicy {
        DISABLED,
        DETECT_ONLY,
        CANCEL_OUTPUT_ON_SPEECH
    }

    public interface Listener {
        void onVadEvent(Event event);
    }

    public static final class Config {
        private final int sampleRateHz;
        private final int calibrationMs;
        private final int activationMs;
        private final int minimumSpeechMs;
        private final int candidateGapMs;
        private final int silenceEndpointMs;
        private final int maxUtteranceMs;
        private final int preRollMs;
        private final double activationSnrDb;
        private final double releaseSnrDb;
        private final double minimumActivationDbfs;
        private final double minimumReleaseDbfs;
        private final double initialNoiseFloorDbfs;
        private final double noiseRiseAlpha;
        private final double noiseFallAlpha;
        private final double maximumCrestDb;
        private final double minimumEnergeticCoverage;
        private final BargeInPolicy bargeInPolicy;

        private Config(Builder builder) {
            sampleRateHz = builder.sampleRateHz;
            calibrationMs = builder.calibrationMs;
            activationMs = builder.activationMs;
            minimumSpeechMs = builder.minimumSpeechMs;
            candidateGapMs = builder.candidateGapMs;
            silenceEndpointMs = builder.silenceEndpointMs;
            maxUtteranceMs = builder.maxUtteranceMs;
            preRollMs = builder.preRollMs;
            activationSnrDb = builder.activationSnrDb;
            releaseSnrDb = builder.releaseSnrDb;
            minimumActivationDbfs = builder.minimumActivationDbfs;
            minimumReleaseDbfs = builder.minimumReleaseDbfs;
            initialNoiseFloorDbfs = builder.initialNoiseFloorDbfs;
            noiseRiseAlpha = builder.noiseRiseAlpha;
            noiseFallAlpha = builder.noiseFallAlpha;
            maximumCrestDb = builder.maximumCrestDb;
            minimumEnergeticCoverage = builder.minimumEnergeticCoverage;
            bargeInPolicy = builder.bargeInPolicy;
            validate();
        }

        public static Builder builder() {
            return new Builder();
        }

        public int getSampleRateHz() {
            return sampleRateHz;
        }

        public int getPreRollSamples() {
            return millisecondsToSamples(preRollMs, sampleRateHz);
        }

        public int getPreRollBytesPcm16Mono() {
            return getPreRollSamples() * 2;
        }

        public int getSilenceEndpointMs() {
            return silenceEndpointMs;
        }

        public int getMaxUtteranceMs() {
            return maxUtteranceMs;
        }

        public BargeInPolicy getBargeInPolicy() {
            return bargeInPolicy;
        }

        private void validate() {
            require(sampleRateHz == SAMPLE_RATE_HZ, "only 16 kHz PCM is supported");
            require(calibrationMs >= 0, "calibrationMs must be non-negative");
            require(activationMs > 0, "activationMs must be positive");
            require(minimumSpeechMs >= activationMs,
                    "minimumSpeechMs must be at least activationMs");
            require(candidateGapMs >= 0, "candidateGapMs must be non-negative");
            require(silenceEndpointMs > candidateGapMs,
                    "silenceEndpointMs must exceed candidateGapMs");
            require(maxUtteranceMs > minimumSpeechMs,
                    "maxUtteranceMs must exceed minimumSpeechMs");
            require(preRollMs >= 0, "preRollMs must be non-negative");
            require(activationSnrDb > releaseSnrDb && releaseSnrDb >= 0.0,
                    "activationSnrDb must exceed non-negative releaseSnrDb");
            require(minimumActivationDbfs > minimumReleaseDbfs,
                    "minimumActivationDbfs must exceed minimumReleaseDbfs");
            require(initialNoiseFloorDbfs >= MIN_DBFS && initialNoiseFloorDbfs < 0.0,
                    "initialNoiseFloorDbfs is outside the supported range");
            require(noiseRiseAlpha > 0.0 && noiseRiseAlpha <= 1.0,
                    "noiseRiseAlpha must be in (0, 1]");
            require(noiseFallAlpha > 0.0 && noiseFallAlpha <= 1.0,
                    "noiseFallAlpha must be in (0, 1]");
            require(maximumCrestDb > 0.0, "maximumCrestDb must be positive");
            require(minimumEnergeticCoverage >= 0.0 && minimumEnergeticCoverage <= 1.0,
                    "minimumEnergeticCoverage must be in [0, 1]");
            require(bargeInPolicy != null, "bargeInPolicy is required");
        }

        public static final class Builder {
            private int sampleRateHz = SAMPLE_RATE_HZ;
            private int calibrationMs = 300;
            private int activationMs = 60;
            private int minimumSpeechMs = 160;
            private int candidateGapMs = 60;
            private int silenceEndpointMs = 420;
            private int maxUtteranceMs = 15_000;
            private int preRollMs = 240;
            private double activationSnrDb = 10.0;
            private double releaseSnrDb = 6.0;
            private double minimumActivationDbfs = -48.0;
            private double minimumReleaseDbfs = -54.0;
            private double initialNoiseFloorDbfs = -64.0;
            private double noiseRiseAlpha = 0.12;
            private double noiseFallAlpha = 0.025;
            private double maximumCrestDb = 20.0;
            private double minimumEnergeticCoverage = 0.06;
            private BargeInPolicy bargeInPolicy = BargeInPolicy.DISABLED;

            public Builder setCalibrationMs(int value) {
                calibrationMs = value;
                return this;
            }

            public Builder setActivationMs(int value) {
                activationMs = value;
                return this;
            }

            public Builder setMinimumSpeechMs(int value) {
                minimumSpeechMs = value;
                return this;
            }

            public Builder setCandidateGapMs(int value) {
                candidateGapMs = value;
                return this;
            }

            public Builder setSilenceEndpointMs(int value) {
                silenceEndpointMs = value;
                return this;
            }

            public Builder setMaxUtteranceMs(int value) {
                maxUtteranceMs = value;
                return this;
            }

            public Builder setPreRollMs(int value) {
                preRollMs = value;
                return this;
            }

            public Builder setActivationSnrDb(double value) {
                activationSnrDb = value;
                return this;
            }

            public Builder setReleaseSnrDb(double value) {
                releaseSnrDb = value;
                return this;
            }

            public Builder setMinimumActivationDbfs(double value) {
                minimumActivationDbfs = value;
                return this;
            }

            public Builder setMinimumReleaseDbfs(double value) {
                minimumReleaseDbfs = value;
                return this;
            }

            public Builder setInitialNoiseFloorDbfs(double value) {
                initialNoiseFloorDbfs = value;
                return this;
            }

            public Builder setNoiseRiseAlpha(double value) {
                noiseRiseAlpha = value;
                return this;
            }

            public Builder setNoiseFallAlpha(double value) {
                noiseFallAlpha = value;
                return this;
            }

            public Builder setMaximumCrestDb(double value) {
                maximumCrestDb = value;
                return this;
            }

            public Builder setMinimumEnergeticCoverage(double value) {
                minimumEnergeticCoverage = value;
                return this;
            }

            public Builder setBargeInPolicy(BargeInPolicy value) {
                bargeInPolicy = value;
                return this;
            }

            public Config build() {
                return new Config(this);
            }
        }
    }

    public static final class Event {
        private final long sequence;
        private final EventType type;
        private final State previousState;
        private final State state;
        private final EndpointReason endpointReason;
        private final long decisionSample;
        private final long speechStartSample;
        private final long speechEndSample;
        private final long recommendedAudioStartSample;
        private final double rmsDbfs;
        private final double noiseFloorDbfs;
        private final double thresholdDbfs;
        private final boolean bargeIn;
        private final boolean cancelOutputRecommended;

        private Event(
                long sequence,
                EventType type,
                State previousState,
                State state,
                EndpointReason endpointReason,
                long decisionSample,
                long speechStartSample,
                long speechEndSample,
                long recommendedAudioStartSample,
                double rmsDbfs,
                double noiseFloorDbfs,
                double thresholdDbfs,
                boolean bargeIn,
                boolean cancelOutputRecommended
        ) {
            this.sequence = sequence;
            this.type = type;
            this.previousState = previousState;
            this.state = state;
            this.endpointReason = endpointReason;
            this.decisionSample = decisionSample;
            this.speechStartSample = speechStartSample;
            this.speechEndSample = speechEndSample;
            this.recommendedAudioStartSample = recommendedAudioStartSample;
            this.rmsDbfs = rmsDbfs;
            this.noiseFloorDbfs = noiseFloorDbfs;
            this.thresholdDbfs = thresholdDbfs;
            this.bargeIn = bargeIn;
            this.cancelOutputRecommended = cancelOutputRecommended;
        }

        public long getSequence() {
            return sequence;
        }

        public EventType getType() {
            return type;
        }

        public State getPreviousState() {
            return previousState;
        }

        public State getState() {
            return state;
        }

        public EndpointReason getEndpointReason() {
            return endpointReason;
        }

        public long getDecisionSample() {
            return decisionSample;
        }

        public long getSpeechStartSample() {
            return speechStartSample;
        }

        public long getSpeechEndSample() {
            return speechEndSample;
        }

        public long getRecommendedAudioStartSample() {
            return recommendedAudioStartSample;
        }

        public double getRmsDbfs() {
            return rmsDbfs;
        }

        public double getNoiseFloorDbfs() {
            return noiseFloorDbfs;
        }

        public double getThresholdDbfs() {
            return thresholdDbfs;
        }

        public boolean isBargeIn() {
            return bargeIn;
        }

        public boolean isCancelOutputRecommended() {
            return cancelOutputRecommended;
        }
    }

    public static final class FrameResult {
        private final State state;
        private final EndpointReason endpointReason;
        private final boolean energyActive;
        private final boolean impulseRejected;
        private final boolean outputBlocked;
        private final double rmsDbfs;
        private final double noiseFloorDbfs;
        private final double thresholdDbfs;

        private FrameResult(
                State state,
                EndpointReason endpointReason,
                boolean energyActive,
                boolean impulseRejected,
                boolean outputBlocked,
                double rmsDbfs,
                double noiseFloorDbfs,
                double thresholdDbfs
        ) {
            this.state = state;
            this.endpointReason = endpointReason;
            this.energyActive = energyActive;
            this.impulseRejected = impulseRejected;
            this.outputBlocked = outputBlocked;
            this.rmsDbfs = rmsDbfs;
            this.noiseFloorDbfs = noiseFloorDbfs;
            this.thresholdDbfs = thresholdDbfs;
        }

        public State getState() {
            return state;
        }

        public EndpointReason getEndpointReason() {
            return endpointReason;
        }

        public boolean isEnergyActive() {
            return energyActive;
        }

        public boolean isImpulseRejected() {
            return impulseRejected;
        }

        public boolean isOutputBlocked() {
            return outputBlocked;
        }

        public double getRmsDbfs() {
            return rmsDbfs;
        }

        public double getNoiseFloorDbfs() {
            return noiseFloorDbfs;
        }

        public double getThresholdDbfs() {
            return thresholdDbfs;
        }
    }

    public static final class Metrics {
        private final long framesProcessed;
        private final long samplesProcessed;
        private final long noiseUpdates;
        private final long confirmedUtterances;
        private final long endpoints;
        private final long trailingSilenceEndpoints;
        private final long maximumDurationEndpoints;
        private final long rejectedShortBursts;
        private final long rejectedImpulseFrames;
        private final long outputBlockedFrames;
        private final long cancellations;
        private final double noiseFloorDbfs;

        private Metrics(AdaptiveEnergyVad vad) {
            framesProcessed = vad.framesProcessed;
            samplesProcessed = vad.samplesProcessed;
            noiseUpdates = vad.noiseUpdates;
            confirmedUtterances = vad.confirmedUtterances;
            endpoints = vad.endpoints;
            trailingSilenceEndpoints = vad.trailingSilenceEndpoints;
            maximumDurationEndpoints = vad.maximumDurationEndpoints;
            rejectedShortBursts = vad.rejectedShortBursts;
            rejectedImpulseFrames = vad.rejectedImpulseFrames;
            outputBlockedFrames = vad.outputBlockedFrames;
            cancellations = vad.cancellations;
            noiseFloorDbfs = vad.noiseFloorDbfs;
        }

        public long getFramesProcessed() {
            return framesProcessed;
        }

        public long getSamplesProcessed() {
            return samplesProcessed;
        }

        public long getNoiseUpdates() {
            return noiseUpdates;
        }

        public long getConfirmedUtterances() {
            return confirmedUtterances;
        }

        public long getEndpoints() {
            return endpoints;
        }

        public long getTrailingSilenceEndpoints() {
            return trailingSilenceEndpoints;
        }

        public long getMaximumDurationEndpoints() {
            return maximumDurationEndpoints;
        }

        public long getRejectedShortBursts() {
            return rejectedShortBursts;
        }

        public long getRejectedImpulseFrames() {
            return rejectedImpulseFrames;
        }

        public long getOutputBlockedFrames() {
            return outputBlockedFrames;
        }

        public long getCancellations() {
            return cancellations;
        }

        public double getNoiseFloorDbfs() {
            return noiseFloorDbfs;
        }
    }

    private final Config config;
    private final Listener listener;
    private final long calibrationSamplesRequired;
    private final long activationSamplesRequired;
    private final long minimumSpeechSamplesRequired;
    private final long candidateGapSamplesAllowed;
    private final long silenceEndpointSamplesRequired;
    private final long maximumUtteranceSamplesAllowed;
    private final long preRollSamples;

    private volatile State state = State.IDLE;
    private volatile boolean assistantOutputActive;
    private long eventSequence;
    private long framesProcessed;
    private long samplesProcessed;
    private long calibratedSamples;
    private long noiseUpdates;
    private long confirmedUtterances;
    private long endpoints;
    private long trailingSilenceEndpoints;
    private long maximumDurationEndpoints;
    private long rejectedShortBursts;
    private long rejectedImpulseFrames;
    private long outputBlockedFrames;
    private long cancellations;
    private double noiseFloorDbfs;
    private double lastRmsDbfs = MIN_DBFS;
    private double lastThresholdDbfs = MIN_DBFS;

    private long candidateStartSample = -1;
    private long candidateActiveSamples;
    private long candidateConsecutiveSamples;
    private long candidateGapSamples;
    private long candidateLastActiveEndSample = -1;
    private boolean candidateActivated;

    private long utteranceStartSample = -1;
    private long lastActiveEndSample = -1;
    private long utteranceActiveSamples;
    private long trailingSilenceSamples;
    private boolean utteranceWasBargeIn;

    public AdaptiveEnergyVad(Config config, Listener listener) {
        if (config == null) {
            throw new IllegalArgumentException("config is required");
        }
        this.config = config;
        this.listener = listener;
        calibrationSamplesRequired = millisecondsToSamples(config.calibrationMs,
                config.sampleRateHz);
        activationSamplesRequired = millisecondsToSamples(config.activationMs,
                config.sampleRateHz);
        minimumSpeechSamplesRequired = millisecondsToSamples(config.minimumSpeechMs,
                config.sampleRateHz);
        candidateGapSamplesAllowed = millisecondsToSamples(config.candidateGapMs,
                config.sampleRateHz);
        silenceEndpointSamplesRequired = millisecondsToSamples(config.silenceEndpointMs,
                config.sampleRateHz);
        maximumUtteranceSamplesAllowed = millisecondsToSamples(config.maxUtteranceMs,
                config.sampleRateHz);
        preRollSamples = millisecondsToSamples(config.preRollMs, config.sampleRateHz);
        noiseFloorDbfs = config.initialNoiseFloorDbfs;
    }

    public AdaptiveEnergyVad(Config config) {
        this(config, null);
    }

    public State getState() {
        return state;
    }

    public Config getConfig() {
        return config;
    }

    public synchronized Metrics snapshotMetrics() {
        return new Metrics(this);
    }

    public void setAssistantOutputActive(boolean active) {
        assistantOutputActive = active;
    }

    public synchronized void start() {
        if (state != State.IDLE) {
            throw new IllegalStateException("start requires IDLE state, found " + state);
        }
        clearUtteranceTracking();
        transition(State.LISTENING, EndpointReason.NONE, samplesProcessed);
    }

    /** Preserves the learned noise floor and cumulative metrics for the next utterance. */
    public synchronized void rearm() {
        if (state != State.ENDPOINT) {
            throw new IllegalStateException("rearm requires ENDPOINT state, found " + state);
        }
        clearUtteranceTracking();
        transition(State.LISTENING, EndpointReason.NONE, samplesProcessed);
    }

    /** Cancels the active listen/utterance without producing an endpoint payload. */
    public synchronized void cancel() {
        if (state == State.IDLE) {
            return;
        }
        State previous = state;
        cancellations++;
        state = State.IDLE;
        emit(EventType.STATE_CHANGED, previous, State.IDLE, EndpointReason.NONE,
                samplesProcessed, -1, -1, -1, false, false);
        emit(EventType.CANCELLED, previous, State.IDLE, EndpointReason.NONE,
                samplesProcessed, -1, -1, -1, false, false);
        clearUtteranceTracking();
    }

    /** Restores construction state, including the initial noise floor and all metrics. */
    public synchronized void reset() {
        state = State.IDLE;
        assistantOutputActive = false;
        eventSequence = 0;
        framesProcessed = 0;
        samplesProcessed = 0;
        calibratedSamples = 0;
        noiseUpdates = 0;
        confirmedUtterances = 0;
        endpoints = 0;
        trailingSilenceEndpoints = 0;
        maximumDurationEndpoints = 0;
        rejectedShortBursts = 0;
        rejectedImpulseFrames = 0;
        outputBlockedFrames = 0;
        cancellations = 0;
        noiseFloorDbfs = config.initialNoiseFloorDbfs;
        lastRmsDbfs = MIN_DBFS;
        lastThresholdDbfs = MIN_DBFS;
        clearUtteranceTracking();
    }

    public FrameResult processFrame(short[] pcm) {
        if (pcm == null) {
            throw new IllegalArgumentException("pcm is required");
        }
        return processFrame(pcm, 0, pcm.length);
    }

    /**
     * Processes one mono PCM16 frame. Ten or twenty millisecond frames are recommended; all timing
     * is sample-count based, so a final short read remains deterministic.
     */
    public synchronized FrameResult processFrame(short[] pcm, int offset, int length) {
        if (state == State.IDLE || state == State.ENDPOINT) {
            throw new IllegalStateException("processFrame requires LISTENING/SPEECH/HANGOVER state");
        }
        if (pcm == null || offset < 0 || length <= 0 || offset > pcm.length - length) {
            throw new IllegalArgumentException("invalid PCM frame range");
        }

        long frameStartSample = samplesProcessed;
        long frameEndSample = frameStartSample + length;
        FrameFeatures features = measure(pcm, offset, length);
        framesProcessed++;
        samplesProcessed = frameEndSample;
        lastRmsDbfs = features.rmsDbfs;

        double activationThreshold = activationThresholdDbfs();
        double releaseThreshold = releaseThresholdDbfs();
        double threshold = (state == State.LISTENING && !candidateActivated)
                ? activationThreshold : releaseThreshold;
        lastThresholdDbfs = threshold;

        boolean outputBlocked = assistantOutputActive
                && config.bargeInPolicy == BargeInPolicy.DISABLED;
        if (outputBlocked) {
            outputBlockedFrames++;
            handleInactiveFrame(length, frameEndSample, false);
            return result(false, false, true, EndpointReason.NONE);
        }

        boolean calibration = state == State.LISTENING
                && calibratedSamples < calibrationSamplesRequired;
        boolean impulse = features.rmsDbfs >= activationThreshold
                && (features.crestDb > config.maximumCrestDb
                || features.energeticCoverage < config.minimumEnergeticCoverage);
        if (impulse) {
            rejectedImpulseFrames++;
        }

        if (calibration) {
            if (!impulse) {
                updateNoiseFloor(features.rmsDbfs);
            }
            calibratedSamples = Math.min(calibrationSamplesRequired,
                    calibratedSamples + length);
            clearCandidate(false, frameEndSample);
            return result(false, impulse, false, EndpointReason.NONE);
        }

        boolean energyActive = !impulse && features.rmsDbfs >= threshold;
        if (state == State.LISTENING) {
            if (energyActive) {
                handleListeningActivity(frameStartSample, frameEndSample, length);
            } else {
                if (!impulse) {
                    updateNoiseFloor(features.rmsDbfs);
                }
                handleInactiveFrame(length, frameEndSample, true);
            }
        } else if (energyActive) {
            utteranceActiveSamples += length;
            lastActiveEndSample = frameEndSample;
            trailingSilenceSamples = 0;
            if (state == State.HANGOVER) {
                transition(State.SPEECH, EndpointReason.NONE, frameEndSample);
            }
        } else {
            handleInactiveFrame(length, frameEndSample, true);
        }

        EndpointReason reason = maybeEndpoint(frameEndSample);
        return result(energyActive, impulse, false, reason);
    }

    private void handleListeningActivity(long frameStartSample, long frameEndSample, int length) {
        if (candidateStartSample < 0) {
            candidateStartSample = frameStartSample;
            candidateLastActiveEndSample = frameEndSample;
        }
        candidateActiveSamples += length;
        candidateConsecutiveSamples += length;
        candidateGapSamples = 0;
        candidateLastActiveEndSample = frameEndSample;
        if (candidateConsecutiveSamples >= activationSamplesRequired) {
            candidateActivated = true;
        }
        if (candidateActivated && candidateActiveSamples >= minimumSpeechSamplesRequired) {
            confirmSpeech(frameEndSample);
        }
    }

    private void handleInactiveFrame(int length, long frameEndSample, boolean countBurst) {
        if (state == State.LISTENING) {
            if (candidateStartSample < 0) {
                return;
            }
            candidateGapSamples += length;
            if (!candidateActivated) {
                clearCandidate(countBurst, frameEndSample);
            } else if (candidateGapSamples > candidateGapSamplesAllowed) {
                clearCandidate(countBurst, frameEndSample);
            }
            return;
        }

        if (state == State.SPEECH) {
            transition(State.HANGOVER, EndpointReason.NONE, frameEndSample);
        }
        if (state == State.HANGOVER) {
            trailingSilenceSamples += length;
        }
    }

    private void confirmSpeech(long decisionSample) {
        utteranceStartSample = candidateStartSample;
        lastActiveEndSample = candidateLastActiveEndSample;
        utteranceActiveSamples = candidateActiveSamples;
        trailingSilenceSamples = 0;
        utteranceWasBargeIn = assistantOutputActive;
        confirmedUtterances++;
        long audioStart = Math.max(0, utteranceStartSample - preRollSamples);
        transition(State.SPEECH, EndpointReason.NONE, decisionSample);
        emit(EventType.SPEECH_CONFIRMED, State.LISTENING, State.SPEECH,
                EndpointReason.NONE, decisionSample, utteranceStartSample, lastActiveEndSample,
                audioStart, utteranceWasBargeIn,
                utteranceWasBargeIn
                        && config.bargeInPolicy == BargeInPolicy.CANCEL_OUTPUT_ON_SPEECH);
        clearCandidateFields();
    }

    private EndpointReason maybeEndpoint(long frameEndSample) {
        if (state != State.SPEECH && state != State.HANGOVER) {
            return EndpointReason.NONE;
        }
        if (frameEndSample - utteranceStartSample >= maximumUtteranceSamplesAllowed) {
            finishEndpoint(EndpointReason.MAX_UTTERANCE, frameEndSample);
            return EndpointReason.MAX_UTTERANCE;
        }
        if (state == State.HANGOVER
                && trailingSilenceSamples >= silenceEndpointSamplesRequired) {
            finishEndpoint(EndpointReason.TRAILING_SILENCE, frameEndSample);
            return EndpointReason.TRAILING_SILENCE;
        }
        return EndpointReason.NONE;
    }

    private void finishEndpoint(EndpointReason reason, long decisionSample) {
        State previous = state;
        endpoints++;
        if (reason == EndpointReason.TRAILING_SILENCE) {
            trailingSilenceEndpoints++;
        } else if (reason == EndpointReason.MAX_UTTERANCE) {
            maximumDurationEndpoints++;
        }
        long audioStart = Math.max(0, utteranceStartSample - preRollSamples);
        transition(State.ENDPOINT, reason, decisionSample);
        emit(EventType.ENDPOINT, previous, State.ENDPOINT, reason, decisionSample,
                utteranceStartSample, lastActiveEndSample, audioStart, utteranceWasBargeIn,
                false);
    }

    private void clearCandidate(boolean countRejection, long decisionSample) {
        if (countRejection && candidateActiveSamples > 0) {
            rejectedShortBursts++;
            emit(EventType.SHORT_BURST_REJECTED, State.LISTENING, State.LISTENING,
                    EndpointReason.NONE, decisionSample, candidateStartSample,
                    candidateLastActiveEndSample,
                    Math.max(0, candidateStartSample - preRollSamples), false, false);
        }
        clearCandidateFields();
    }

    private void clearCandidateFields() {
        candidateStartSample = -1;
        candidateActiveSamples = 0;
        candidateConsecutiveSamples = 0;
        candidateGapSamples = 0;
        candidateLastActiveEndSample = -1;
        candidateActivated = false;
    }

    private void clearUtteranceTracking() {
        clearCandidateFields();
        utteranceStartSample = -1;
        lastActiveEndSample = -1;
        utteranceActiveSamples = 0;
        trailingSilenceSamples = 0;
        utteranceWasBargeIn = false;
    }

    private void transition(State next, EndpointReason reason, long decisionSample) {
        if (state == next) {
            return;
        }
        State previous = state;
        state = next;
        emit(EventType.STATE_CHANGED, previous, next, reason, decisionSample,
                utteranceStartSample, lastActiveEndSample,
                utteranceStartSample < 0 ? -1 : Math.max(0, utteranceStartSample - preRollSamples),
                utteranceWasBargeIn, false);
    }

    private void emit(
            EventType type,
            State previous,
            State next,
            EndpointReason reason,
            long decisionSample,
            long speechStart,
            long speechEnd,
            long audioStart,
            boolean bargeIn,
            boolean cancelOutput
    ) {
        if (listener == null) {
            return;
        }
        listener.onVadEvent(new Event(
                ++eventSequence,
                type,
                previous,
                next,
                reason,
                decisionSample,
                speechStart,
                speechEnd,
                audioStart,
                lastRmsDbfs,
                noiseFloorDbfs,
                lastThresholdDbfs,
                bargeIn,
                cancelOutput
        ));
    }

    private FrameResult result(
            boolean energyActive,
            boolean impulse,
            boolean outputBlocked,
            EndpointReason reason
    ) {
        return new FrameResult(state, reason, energyActive, impulse, outputBlocked,
                lastRmsDbfs, noiseFloorDbfs, lastThresholdDbfs);
    }

    private double activationThresholdDbfs() {
        return Math.max(config.minimumActivationDbfs,
                noiseFloorDbfs + config.activationSnrDb);
    }

    private double releaseThresholdDbfs() {
        return Math.max(config.minimumReleaseDbfs,
                noiseFloorDbfs + config.releaseSnrDb);
    }

    private void updateNoiseFloor(double observedDbfs) {
        double clamped = clamp(observedDbfs, MIN_DBFS, -15.0);
        double alpha = clamped > noiseFloorDbfs
                ? config.noiseRiseAlpha : config.noiseFallAlpha;
        noiseFloorDbfs += alpha * (clamped - noiseFloorDbfs);
        noiseFloorDbfs = clamp(noiseFloorDbfs, MIN_DBFS, -15.0);
        noiseUpdates++;
    }

    private static FrameFeatures measure(short[] pcm, int offset, int length) {
        double sumSquares = 0.0;
        int peak = 0;
        int limit = offset + length;
        for (int index = offset; index < limit; index++) {
            int sample = pcm[index];
            int magnitude = sample == Short.MIN_VALUE ? 32_768 : Math.abs(sample);
            sumSquares += (double) sample * sample;
            if (magnitude > peak) {
                peak = magnitude;
            }
        }
        double rms = Math.sqrt(sumSquares / length);
        double rmsDbfs = amplitudeToDbfs(rms);
        double peakDbfs = amplitudeToDbfs(peak);
        double crestDb = rms > 0.0 ? peakDbfs - rmsDbfs : 0.0;
        double energeticFloor = Math.max(64.0, rms * 0.35);
        int energeticSamples = 0;
        for (int index = offset; index < limit; index++) {
            int sample = pcm[index];
            int magnitude = sample == Short.MIN_VALUE ? 32_768 : Math.abs(sample);
            if (magnitude >= energeticFloor) {
                energeticSamples++;
            }
        }
        return new FrameFeatures(rmsDbfs, crestDb,
                energeticSamples / (double) length);
    }

    private static double amplitudeToDbfs(double amplitude) {
        if (amplitude <= 0.0) {
            return MIN_DBFS;
        }
        return Math.max(MIN_DBFS, 20.0 * Math.log10(amplitude / PCM16_SCALE));
    }

    private static int millisecondsToSamples(int milliseconds, int sampleRateHz) {
        long samples = ((long) milliseconds * sampleRateHz + 999L) / 1000L;
        if (samples > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("duration is too large");
        }
        return (int) samples;
    }

    private static double clamp(double value, double low, double high) {
        return Math.max(low, Math.min(high, value));
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }

    private static final class FrameFeatures {
        private final double rmsDbfs;
        private final double crestDb;
        private final double energeticCoverage;

        private FrameFeatures(double rmsDbfs, double crestDb, double energeticCoverage) {
            this.rmsDbfs = rmsDbfs;
            this.crestDb = crestDb;
            this.energeticCoverage = energeticCoverage;
        }
    }
}
