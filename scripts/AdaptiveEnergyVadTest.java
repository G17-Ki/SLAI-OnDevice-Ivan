package ai.liquid.lfmdemo;

import java.util.ArrayList;
import java.util.List;

/** Dependency-free host tests for AdaptiveEnergyVad. */
public final class AdaptiveEnergyVadTest {
    private static final int FRAME_SAMPLES = 320;
    private static long waveformSample;
    private static int noiseState = 0x13579bdf;

    public static void main(String[] args) {
        testCleanSpeechAndPreRoll();
        testBackgroundNoiseAdaptation();
        testClickAndShortBurstRejection();
        testTrailingSilenceEndpoint();
        testMaximumDurationEndpoint();
        testCancelAndReset();
        testTwoConsecutiveUtterances();
        testBargeInPolicies();
        System.out.println("AdaptiveEnergyVad: 8 tests passed");
    }

    private static void testCleanSpeechAndPreRoll() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad vad = newVad(sink, baseBuilder());
        vad.start();
        feed(vad, silence(), 10);
        double floorBeforeSpeech = vad.snapshotMetrics().getNoiseFloorDbfs();
        feed(vad, tone(6_000), 10);

        check(vad.getState() == AdaptiveEnergyVad.State.SPEECH,
                "clean speech should be confirmed");
        AdaptiveEnergyVad.Event speech = sink.only(AdaptiveEnergyVad.EventType.SPEECH_CONFIRMED);
        check(speech.getSpeechStartSample() == 10L * FRAME_SAMPLES,
                "speech start must use the first provisional frame");
        check(speech.getRecommendedAudioStartSample() == 0,
                "240 ms pre-roll should reach capture start in this fixture");
        check(speech.getDecisionSample() - speech.getSpeechStartSample() == 8L * FRAME_SAMPLES,
                "minimum-speech confirmation should be sample deterministic");
        check(vad.snapshotMetrics().getNoiseFloorDbfs() == floorBeforeSpeech,
                "provisional and confirmed speech must not update the noise floor");
        check(strictlyIncreasingSequences(sink.events), "event sequence must be monotonic");
    }

    private static void testBackgroundNoiseAdaptation() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad.Config.Builder builder = baseBuilder()
                .setInitialNoiseFloorDbfs(-70.0)
                .setMinimumActivationDbfs(-55.0)
                .setMinimumReleaseDbfs(-60.0)
                .setActivationSnrDb(12.0)
                .setReleaseSnrDb(7.0)
                .setNoiseRiseAlpha(0.20);
        AdaptiveEnergyVad vad = newVad(sink, builder);
        vad.start();
        int[] gradualAmbient = {80, 120, 180, 260, 380, 550, 800};
        for (int amplitude : gradualAmbient) {
            feed(vad, noise(amplitude), 40);
        }

        AdaptiveEnergyVad.Metrics metrics = vad.snapshotMetrics();
        check(vad.getState() == AdaptiveEnergyVad.State.LISTENING,
                "gradually rising stationary noise must not become speech");
        check(metrics.getConfirmedUtterances() == 0, "noise produced an utterance");
        check(metrics.getNoiseFloorDbfs() > -40.0,
                "noise floor did not adapt upward: " + metrics.getNoiseFloorDbfs());
        check(metrics.getNoiseUpdates() == metrics.getFramesProcessed(),
                "all inactive ambient frames should update the floor");
    }

    private static void testClickAndShortBurstRejection() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad vad = newVad(sink, baseBuilder());
        vad.start();
        short[] click = silence();
        click[FRAME_SAMPLES / 2] = Short.MAX_VALUE;
        vad.processFrame(click);
        feed(vad, tone(7_000), 3);
        feed(vad, silence(), 5);

        AdaptiveEnergyVad.Metrics metrics = vad.snapshotMetrics();
        check(vad.getState() == AdaptiveEnergyVad.State.LISTENING,
                "click/short burst must leave VAD listening");
        check(metrics.getRejectedImpulseFrames() == 1, "click was not rejected");
        check(metrics.getRejectedShortBursts() == 1, "short burst was not rejected");
        check(metrics.getConfirmedUtterances() == 0, "burst became confirmed speech");
        check(sink.count(AdaptiveEnergyVad.EventType.ENDPOINT) == 0,
                "rejected burst must not generate a request endpoint");
    }

    private static void testTrailingSilenceEndpoint() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad vad = newVad(sink, baseBuilder());
        vad.start();
        feed(vad, tone(5_000), 12);
        double floorBeforeHangover = vad.snapshotMetrics().getNoiseFloorDbfs();
        feedUntilEndpoint(vad, silence(), 30);

        check(vad.getState() == AdaptiveEnergyVad.State.ENDPOINT,
                "trailing silence must close the utterance");
        AdaptiveEnergyVad.Event endpoint = sink.only(AdaptiveEnergyVad.EventType.ENDPOINT);
        check(endpoint.getEndpointReason() == AdaptiveEnergyVad.EndpointReason.TRAILING_SILENCE,
                "wrong trailing endpoint reason");
        check(endpoint.getDecisionSample() - endpoint.getSpeechEndSample()
                        == samplesForMs(420),
                "endpoint hangover must equal configured 420 ms");
        check(vad.snapshotMetrics().getTrailingSilenceEndpoints() == 1,
                "trailing endpoint metric missing");
        check(vad.snapshotMetrics().getNoiseFloorDbfs() == floorBeforeHangover,
                "confirmed hangover must not update the noise floor");
    }

    private static void testMaximumDurationEndpoint() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad.Config.Builder builder = baseBuilder()
                .setMaximumCrestDb(22.0)
                .setMaxUtteranceMs(600);
        AdaptiveEnergyVad vad = newVad(sink, builder);
        vad.start();
        feedUntilEndpoint(vad, tone(4_000), 50);

        AdaptiveEnergyVad.Event endpoint = sink.only(AdaptiveEnergyVad.EventType.ENDPOINT);
        check(endpoint.getEndpointReason() == AdaptiveEnergyVad.EndpointReason.MAX_UTTERANCE,
                "wrong max-duration endpoint reason");
        check(endpoint.getDecisionSample() - endpoint.getSpeechStartSample()
                        == samplesForMs(600),
                "max-duration endpoint must be sample exact");
        check(vad.snapshotMetrics().getMaximumDurationEndpoints() == 1,
                "max-duration metric missing");
    }

    private static void testCancelAndReset() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad vad = newVad(sink, baseBuilder());
        vad.start();
        feed(vad, tone(5_000), 10);
        vad.cancel();
        check(vad.getState() == AdaptiveEnergyVad.State.IDLE, "cancel must return to IDLE");
        check(vad.snapshotMetrics().getCancellations() == 1, "cancel metric missing");
        check(sink.count(AdaptiveEnergyVad.EventType.CANCELLED) == 1,
                "cancel callback missing");
        vad.reset();
        AdaptiveEnergyVad.Metrics reset = vad.snapshotMetrics();
        check(reset.getFramesProcessed() == 0 && reset.getConfirmedUtterances() == 0
                        && reset.getCancellations() == 0,
                "reset must clear timeline and metrics");
        vad.start();
        check(vad.getState() == AdaptiveEnergyVad.State.LISTENING,
                "VAD must be reusable after reset");
    }

    private static void testTwoConsecutiveUtterances() {
        EventSink sink = new EventSink();
        AdaptiveEnergyVad vad = newVad(sink, baseBuilder());
        vad.start();
        produceUtterance(vad);
        vad.rearm();
        feed(vad, silence(), 5);
        produceUtterance(vad);

        AdaptiveEnergyVad.Metrics metrics = vad.snapshotMetrics();
        check(metrics.getConfirmedUtterances() == 2, "expected two confirmed utterances");
        check(metrics.getEndpoints() == 2, "expected two endpoints");
        check(sink.count(AdaptiveEnergyVad.EventType.SPEECH_CONFIRMED) == 2,
                "expected two speech callbacks");
        check(sink.count(AdaptiveEnergyVad.EventType.ENDPOINT) == 2,
                "expected two endpoint callbacks");
    }

    private static void testBargeInPolicies() {
        EventSink blockedSink = new EventSink();
        AdaptiveEnergyVad blocked = newVad(blockedSink, baseBuilder()
                .setBargeInPolicy(AdaptiveEnergyVad.BargeInPolicy.DISABLED));
        blocked.start();
        blocked.setAssistantOutputActive(true);
        feed(blocked, tone(7_000), 12);
        check(blocked.getState() == AdaptiveEnergyVad.State.LISTENING,
                "disabled barge-in must block output leakage");
        check(blocked.snapshotMetrics().getOutputBlockedFrames() == 12,
                "blocked output frames were not counted");

        EventSink cancelSink = new EventSink();
        AdaptiveEnergyVad cancelling = newVad(cancelSink, baseBuilder()
                .setBargeInPolicy(AdaptiveEnergyVad.BargeInPolicy.CANCEL_OUTPUT_ON_SPEECH));
        cancelling.start();
        cancelling.setAssistantOutputActive(true);
        feed(cancelling, tone(7_000), 10);
        AdaptiveEnergyVad.Event speech = cancelSink.only(
                AdaptiveEnergyVad.EventType.SPEECH_CONFIRMED);
        check(speech.isBargeIn(), "barge-in flag missing");
        check(speech.isCancelOutputRecommended(), "cancel-output recommendation missing");
    }

    private static AdaptiveEnergyVad.Config.Builder baseBuilder() {
        return AdaptiveEnergyVad.Config.builder()
                .setCalibrationMs(0)
                .setActivationMs(60)
                .setMinimumSpeechMs(160)
                .setCandidateGapMs(60)
                .setSilenceEndpointMs(420)
                .setMaxUtteranceMs(3_000)
                .setPreRollMs(240)
                .setInitialNoiseFloorDbfs(-64.0);
    }

    private static AdaptiveEnergyVad newVad(
            EventSink sink,
            AdaptiveEnergyVad.Config.Builder builder
    ) {
        waveformSample = 0;
        return new AdaptiveEnergyVad(builder.build(), sink);
    }

    private static void produceUtterance(AdaptiveEnergyVad vad) {
        feed(vad, tone(5_000), 12);
        feedUntilEndpoint(vad, silence(), 30);
        check(vad.getState() == AdaptiveEnergyVad.State.ENDPOINT,
                "fixture utterance did not endpoint");
    }

    private static void feed(AdaptiveEnergyVad vad, short[] frame, int count) {
        for (int index = 0; index < count; index++) {
            vad.processFrame(frame);
        }
    }

    private static void feedUntilEndpoint(AdaptiveEnergyVad vad, short[] frame, int maximum) {
        for (int index = 0; index < maximum
                && vad.getState() != AdaptiveEnergyVad.State.ENDPOINT; index++) {
            vad.processFrame(frame);
        }
    }

    private static short[] silence() {
        return new short[FRAME_SAMPLES];
    }

    private static short[] tone(int amplitude) {
        short[] frame = new short[FRAME_SAMPLES];
        for (int index = 0; index < frame.length; index++) {
            double phase = 2.0 * Math.PI * 220.0 * waveformSample
                    / AdaptiveEnergyVad.SAMPLE_RATE_HZ;
            frame[index] = (short) Math.round(amplitude * Math.sin(phase));
            waveformSample++;
        }
        return frame;
    }

    private static short[] noise(int amplitude) {
        short[] frame = new short[FRAME_SAMPLES];
        for (int index = 0; index < frame.length; index++) {
            noiseState = noiseState * 1103515245 + 12345;
            int centered = ((noiseState >>> 16) & 0xffff) - 32_768;
            frame[index] = (short) ((long) centered * amplitude / 32_768L);
        }
        return frame;
    }

    private static long samplesForMs(int milliseconds) {
        return (long) AdaptiveEnergyVad.SAMPLE_RATE_HZ * milliseconds / 1_000L;
    }

    private static boolean strictlyIncreasingSequences(List<AdaptiveEnergyVad.Event> events) {
        long previous = 0;
        for (AdaptiveEnergyVad.Event event : events) {
            if (event.getSequence() <= previous) {
                return false;
            }
            previous = event.getSequence();
        }
        return true;
    }

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static final class EventSink implements AdaptiveEnergyVad.Listener {
        private final List<AdaptiveEnergyVad.Event> events = new ArrayList<>();

        @Override
        public void onVadEvent(AdaptiveEnergyVad.Event event) {
            events.add(event);
        }

        private int count(AdaptiveEnergyVad.EventType type) {
            int count = 0;
            for (AdaptiveEnergyVad.Event event : events) {
                if (event.getType() == type) {
                    count++;
                }
            }
            return count;
        }

        private AdaptiveEnergyVad.Event only(AdaptiveEnergyVad.EventType type) {
            AdaptiveEnergyVad.Event found = null;
            for (AdaptiveEnergyVad.Event event : events) {
                if (event.getType() == type) {
                    check(found == null, "more than one " + type + " event");
                    found = event;
                }
            }
            check(found != null, "missing " + type + " event");
            return found;
        }
    }
}
