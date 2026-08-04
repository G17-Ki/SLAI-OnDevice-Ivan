package ai.liquid.lfmdemo;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

final class LocalUiServer {
    private static final String TAG = "LfmUiServer";
    private static final int PORT = 8765;
    private static final Map<String, String> ASSETS = Map.of(
            "/", "demo-ui/index.html",
            "/index.html", "demo-ui/index.html",
            "/manifest.webmanifest", "demo-ui/manifest.webmanifest",
            "/icon.svg", "demo-ui/icon.svg",
            "/service-worker.js", "demo-ui/service-worker.js"
    );
    private static final Map<String, String> CONTENT_TYPES = Map.of(
            "/", "text/html; charset=utf-8",
            "/index.html", "text/html; charset=utf-8",
            "/manifest.webmanifest", "application/manifest+json",
            "/icon.svg", "image/svg+xml",
            "/service-worker.js", "application/javascript; charset=utf-8"
    );

    private static LocalUiServer instance;

    static synchronized void ensureStarted(Context context) {
        if (instance != null) {
            return;
        }
        instance = new LocalUiServer(context.getApplicationContext());
        instance.start();
    }

    private final AssetManager assets;
    private final ExecutorService clients = Executors.newCachedThreadPool();
    private ServerSocket serverSocket;

    private LocalUiServer(Context context) {
        this.assets = context.getAssets();
    }

    private void start() {
        Thread thread = new Thread(() -> {
            try {
                serverSocket = new ServerSocket();
                serverSocket.setReuseAddress(true);
                serverSocket.bind(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), PORT));
                Log.i(TAG, "UI ready on http://127.0.0.1:" + PORT);
                while (!Thread.currentThread().isInterrupted()) {
                    Socket socket = serverSocket.accept();
                    clients.submit(() -> handle(socket));
                }
            } catch (IOException error) {
                Log.e(TAG, "UI server stopped", error);
            }
        }, "lfm-ui-server");
        thread.setDaemon(true);
        thread.start();
    }

    private void handle(Socket socket) {
        try (Socket client = socket;
             BufferedReader reader = new BufferedReader(new InputStreamReader(client.getInputStream(), StandardCharsets.US_ASCII));
             OutputStream output = client.getOutputStream()) {
            String requestLine = reader.readLine();
            if (requestLine == null || requestLine.isEmpty()) {
                return;
            }
            String header;
            while ((header = reader.readLine()) != null && !header.isEmpty()) {
                // Consume request headers.
            }
            String[] parts = requestLine.split(" ");
            String method = parts.length > 0 ? parts[0] : "GET";
            String path = parts.length > 1 ? parts[1].split("\\?", 2)[0] : "/";
            String asset = ASSETS.get(path);
            if (asset == null) {
                writeHeaders(output, "404 Not Found", "text/plain", 0);
                return;
            }
            byte[] body = readAsset(asset);
            writeHeaders(output, "200 OK", CONTENT_TYPES.get(path), body.length);
            if (!"HEAD".equals(method)) {
                output.write(body);
            }
            output.flush();
        } catch (IOException error) {
            Log.w(TAG, "UI request failed", error);
        }
    }

    private byte[] readAsset(String name) throws IOException {
        try (InputStream input = assets.open(name);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static void writeHeaders(OutputStream output, String status, String type, int length) throws IOException {
        String headers = "HTTP/1.1 " + status + "\r\n"
                + "Content-Type: " + type + "\r\n"
                + "Content-Length: " + length + "\r\n"
                + "Cache-Control: no-cache\r\n"
                + "Connection: close\r\n\r\n";
        output.write(headers.getBytes(StandardCharsets.US_ASCII));
    }
}
