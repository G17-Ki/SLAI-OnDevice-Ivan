package ai.liquid.lfm.bootcfg;

import android.app.AlarmManager;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.os.IBinder;
import android.provider.Settings;
import android.util.Log;

public class BootService extends Service {
    public static final long INTERVAL_MS = 30_000L;
    private static final String CHANNEL = "bootcfg";
    private static final int NOTIF_ID = 1;

    @Override
    public void onCreate() {
        super.onCreate();
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        if (nm != null && Build.VERSION.SDK_INT >= 26) {
            NotificationChannel channel = new NotificationChannel(CHANNEL,
                    "BootCfg", NotificationManager.IMPORTANCE_MIN);
            nm.createNotificationChannel(channel);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        startForeground(NOTIF_ID, buildNotification());
        applyTimeout(this);
        scheduleNext(this);
        stopForeground(true);
        stopSelf();
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    public static void applyTimeout(Context context) {
        try {
            Settings.System.putLong(context.getContentResolver(),
                    Settings.System.SCREEN_OFF_TIMEOUT, 1_800_000L);
            Log.i("BootCfg", "re-applied screen_off_timeout=1800000");
        } catch (Exception error) {
            Log.w("BootCfg", "re-apply failed: " + error);
        }
    }

    private Notification buildNotification() {
        Notification.Builder builder = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_bootcfg)
                .setContentTitle("BootCfg")
                .setContentText("30m screen timeout active")
                .setOngoing(true);
        return builder.build();
    }

    public static void scheduleNext(Context context) {
        AlarmManager am = (AlarmManager) context.getSystemService(Context.ALARM_SERVICE);
        if (am == null) return;
        Intent serviceIntent = new Intent(context, BootService.class);
        int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        PendingIntent pi = Build.VERSION.SDK_INT >= 29
                ? PendingIntent.getForegroundService(context, 0, serviceIntent, flags)
                : PendingIntent.getService(context, 0, serviceIntent, flags);
        long triggerAt = System.currentTimeMillis() + INTERVAL_MS;
        try {
            am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAt, pi);
        } catch (SecurityException error) {
            am.set(AlarmManager.RTC_WAKEUP, triggerAt, pi);
        }
    }
}
