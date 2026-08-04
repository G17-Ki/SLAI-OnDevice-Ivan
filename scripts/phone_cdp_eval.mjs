#!/usr/bin/env node

const port = Number(process.argv[2] || 9224);
const rawExpression = process.argv.slice(3).join(" ");
const expression = rawExpression.startsWith("base64:")
  ? Buffer.from(rawExpression.slice(7), "base64").toString("utf8")
  : rawExpression;
if (!expression) {
  console.error("usage: phone_cdp_eval.mjs PORT JAVASCRIPT");
  process.exit(2);
}

const pages = await fetch(`http://127.0.0.1:${port}/json/list`).then(response => {
  if (!response.ok) throw new Error(`CDP page list failed: ${response.status}`);
  return response.json();
});
const page = pages.find(candidate => candidate.type === "page" && candidate.webSocketDebuggerUrl);
if (!page) throw new Error("No debuggable WebView page found");

const socket = new WebSocket(page.webSocketDebuggerUrl);
let nextId = 1;
const pending = new Map();

socket.addEventListener("message", event => {
  const message = JSON.parse(String(event.data));
  if (!message.id) return;
  const waiter = pending.get(message.id);
  if (!waiter) return;
  pending.delete(message.id);
  if (message.error) waiter.reject(new Error(JSON.stringify(message.error)));
  else waiter.resolve(message.result);
});

await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});

function call(method, params = {}) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

await call("Runtime.enable");
const result = await call("Runtime.evaluate", {
  expression,
  awaitPromise: true,
  returnByValue: true,
  userGesture: true,
});
if (result.exceptionDetails) {
  throw new Error(result.exceptionDetails.exception?.description || "Runtime.evaluate failed");
}
console.log(JSON.stringify(result.result?.value, null, 2));
socket.close();
