package ai.onnxruntime;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

/**
 * Exact UINT16 tensor construction for the pinned ORT 1.25 Android binding.
 *
 * <p>ORT 1.25 exposes UINT16 in {@link TensorInfo.OnnxTensorType}, but omits it
 * from {@link OnnxJavaType}. Calling the public ShortBuffer overload would
 * silently create INT16, which is not compatible with the release graph. This
 * helper invokes ORT's own private JNI buffer constructor with ONNX type 4 and
 * then uses package-private handles/constructors from the exact pinned AAR.
 * Any API drift fails closed during class initialization or tensor creation.</p>
 */
public final class UnsignedTensorFactory {
    private static final int UINT16_BYTES = 2;
    private static final Method CREATE_TENSOR_FROM_BUFFER = resolveFactory();

    private UnsignedTensorFactory() {}

    private static Method resolveFactory() {
        try {
            Method method = OnnxTensor.class.getDeclaredMethod(
                    "createTensorFromBuffer",
                    long.class,
                    long.class,
                    java.nio.Buffer.class,
                    int.class,
                    long.class,
                    long[].class,
                    int.class
            );
            method.setAccessible(true);
            return method;
        } catch (ReflectiveOperationException | RuntimeException error) {
            throw new ExceptionInInitializerError(error);
        }
    }

    public static OnnxTensor createUInt16(
            OrtEnvironment environment,
            ShortBuffer source,
            long[] shape
    ) throws OrtException {
        if (environment == null || source == null || shape == null) {
            throw new IllegalArgumentException("environment, source, and shape are required");
        }
        long elements = elementCount(shape);
        if (source.remaining() != elements) {
            throw new IllegalArgumentException(
                    "UINT16 buffer has " + source.remaining() + " elements; expected " + elements
            );
        }

        ShortBuffer direct = source;
        boolean copied = false;
        if (!source.isDirect()) {
            if (elements > Integer.MAX_VALUE / UINT16_BYTES) {
                throw new IllegalArgumentException("UINT16 tensor is too large for a Java buffer");
            }
            ByteBuffer bytes = ByteBuffer
                    .allocateDirect(Math.toIntExact(elements * UINT16_BYTES))
                    .order(ByteOrder.nativeOrder());
            direct = bytes.asShortBuffer();
            int originalPosition = source.position();
            direct.put(source);
            source.position(originalPosition);
            direct.flip();
            copied = true;
        }

        int byteOffset = Math.multiplyExact(direct.position(), UINT16_BYTES);
        long byteSize = Math.multiplyExact((long) direct.remaining(), UINT16_BYTES);
        long handle;
        try {
            handle = ((Number) CREATE_TENSOR_FROM_BUFFER.invoke(
                    null,
                    OnnxRuntime.ortApiHandle,
                    environment.defaultAllocator.handle,
                    direct,
                    byteOffset,
                    byteSize,
                    shape,
                    TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16.value
            )).longValue();
        } catch (IllegalAccessException error) {
            throw new OrtException("ORT UINT16 JNI factory is inaccessible: " + error);
        } catch (InvocationTargetException error) {
            Throwable cause = error.getCause();
            if (cause instanceof OrtException) {
                throw (OrtException) cause;
            }
            throw new OrtException("ORT UINT16 JNI factory failed: " + cause);
        }

        TensorInfo info = new TensorInfo(
                shape.clone(),
                OnnxJavaType.UNKNOWN,
                TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
        );
        return new OnnxTensor(
                handle,
                environment.defaultAllocator.handle,
                info,
                direct,
                copied
        );
    }

    public static OnnxTensor createUInt16Filled(
            OrtEnvironment environment,
            long[] shape,
            int unsignedValue
    ) throws OrtException {
        if (unsignedValue < 0 || unsignedValue > 0xffff) {
            throw new IllegalArgumentException("UINT16 fill is outside [0, 65535]");
        }
        long elements = elementCount(shape);
        if (elements > Integer.MAX_VALUE / UINT16_BYTES) {
            throw new IllegalArgumentException("UINT16 tensor is too large for a Java buffer");
        }
        ShortBuffer values = ByteBuffer
                .allocateDirect(Math.toIntExact(elements * UINT16_BYTES))
                .order(ByteOrder.nativeOrder())
                .asShortBuffer();
        short encoded = (short) unsignedValue;
        while (values.hasRemaining()) {
            values.put(encoded);
        }
        values.flip();
        return createUInt16(environment, values, shape);
    }

    private static long elementCount(long[] shape) {
        long elements = 1;
        for (long dimension : shape) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("UINT16 shape must be fully static and positive");
            }
            elements = Math.multiplyExact(elements, dimension);
        }
        return elements;
    }
}
