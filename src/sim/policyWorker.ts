import * as ort from "onnxruntime-web/wasm";
import ortWasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import ortWasmBinaryUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

interface InitRequest {
  id: number;
  type: "init";
  onnxUrl: string;
}

interface InferRequest {
  id: number;
  type: "infer";
  observation: Float32Array;
}

interface DisposeRequest {
  id: number;
  type: "dispose";
}

type WorkerRequest = InitRequest | InferRequest | DisposeRequest;

ort.env.wasm.wasmPaths = {
  mjs: ortWasmModuleUrl,
  wasm: ortWasmBinaryUrl,
};
ort.env.wasm.numThreads = 1;

let session: ort.InferenceSession | null = null;
let inputName = "obs";
let outputName = "actions";

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  void handleRequest(event.data);
};

async function handleRequest(request: WorkerRequest): Promise<void> {
  try {
    if (request.type === "init") {
      await session?.release();
      session = await ort.InferenceSession.create(request.onnxUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      inputName = session.inputNames[0] ?? "obs";
      outputName = session.outputNames[0] ?? "actions";
      postSuccess(request.id, { inputName, outputName });
      return;
    }

    if (request.type === "dispose") {
      await session?.release();
      session = null;
      postSuccess(request.id, {});
      return;
    }

    if (!session) {
      throw new Error("Policy worker is not initialized");
    }

    const inputTensor = new ort.Tensor("float32", request.observation, [
      1,
      request.observation.length,
    ]);
    const started = performance.now();
    const outputs = await session.run({ [inputName]: inputTensor });
    const inferenceMs = performance.now() - started;
    const output = outputs[outputName];
    if (!output) {
      throw new Error(`ONNX output not found: ${outputName}`);
    }

    const action = new Float32Array(output.data as Float32Array | number[]);
    postSuccess(request.id, { action, inferenceMs }, [action.buffer]);
  } catch (error) {
    postFailure(request.id, error instanceof Error ? error.message : String(error));
  }
}

function postSuccess(id: number, payload: unknown, transfer: Transferable[] = []): void {
  self.postMessage({ id, ok: true, payload }, { transfer });
}

function postFailure(id: number, error: string): void {
  self.postMessage({ id, ok: false, error });
}
