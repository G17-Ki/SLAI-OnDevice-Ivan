package ai.liquid.lfmdemo.whisperprobe;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Verifies the live ORT session boundary against release metadata exactly. */
public final class ContractVerifier {
    private ContractVerifier() {}

    public static JSONObject verify(
            OrtSession session,
            JSONObject modelMetadata,
            String modelFilename
    ) throws OrtException, JSONException {
        JSONObject model = modelMetadata
                .getJSONObject("model_files")
                .getJSONObject(modelFilename);
        JSONObject result = new JSONObject();
        result.put("inputs", verifySide(session.getInputInfo(), model.getJSONObject("inputs")));
        result.put("outputs", verifySide(session.getOutputInfo(), model.getJSONObject("outputs")));
        return result;
    }

    private static JSONArray verifySide(
            Map<String, NodeInfo> actual,
            JSONObject expected
    ) throws JSONException {
        Set<String> expectedNames = new HashSet<>();
        expected.keys().forEachRemaining(expectedNames::add);
        if (!actual.keySet().equals(expectedNames)) {
            throw new IllegalStateException(
                    "tensor name mismatch; actual=" + actual.keySet() + ", expected=" + expectedNames
            );
        }

        List<String> names = new ArrayList<>(expectedNames);
        Collections.sort(names);
        JSONArray records = new JSONArray();
        for (String name : names) {
            NodeInfo node = actual.get(name);
            if (!(node.getInfo() instanceof TensorInfo)) {
                throw new IllegalStateException("non-tensor boundary: " + name);
            }
            TensorInfo tensor = (TensorInfo) node.getInfo();
            JSONObject expectedTensor = expected.getJSONObject(name);
            String expectedDtype = expectedTensor.getString("dtype");
            String actualDtype = dtype(tensor.onnxType);
            if (!expectedDtype.equals(actualDtype)) {
                throw new IllegalStateException(
                        name + " dtype mismatch; actual=" + actualDtype + ", expected=" + expectedDtype
                );
            }
            long[] expectedShape = jsonShape(expectedTensor.getJSONArray("shape"));
            long[] actualShape = tensor.getShape();
            if (!Arrays.equals(actualShape, expectedShape)) {
                throw new IllegalStateException(
                        name + " shape mismatch; actual=" + Arrays.toString(actualShape)
                                + ", expected=" + Arrays.toString(expectedShape)
                );
            }
            records.put(new JSONObject()
                    .put("name", name)
                    .put("dtype", actualDtype)
                    .put("shape", new JSONArray(actualShape)));
        }
        return records;
    }

    private static long[] jsonShape(JSONArray shape) throws JSONException {
        long[] values = new long[shape.length()];
        for (int index = 0; index < values.length; index++) {
            values[index] = shape.getLong(index);
        }
        return values;
    }

    private static String dtype(TensorInfo.OnnxTensorType type) {
        switch (type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                return "uint8";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
                return "uint16";
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                return "int32";
            default:
                return type.name();
        }
    }
}
