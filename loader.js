const NUM_TIMESTEPS = 50;

WebAssembly.instantiateStreaming(fetch("main.wasm"), {
  env: {},
}).then(({ instance }) => {
  const init = instance.exports.init;
  const step = instance.exports.step;
  const get_sample_ptr = instance.exports.get_sample_ptr;
  const get_sample_len = instance.exports.get_sample_len;

  const memory = new Float32Array(instance.exports.memory.buffer);
  // divide by 4 for Float32Array indexing
  const sample_ptr = get_sample_ptr() / 4;
  const sample_len = get_sample_len();

  const sample = memory.subarray(sample_ptr, sample_ptr + sample_len);

  const xs = new Float32Array(Math.floor(sample.length / 2));
  const ys = new Float32Array(Math.floor(sample.length / 2));

  let t = NUM_TIMESTEPS;

  function updatePlot() {
    for (let i = 0; i < sample.length; i += 2) {
      xs[i / 2] = sample[i];
      ys[i / 2] = sample[i + 1];
    }

    plot(xs, ys, t);
  }

  init();
  step(t);
  updatePlot();

  const stepButton = document.getElementById("step-button");
  stepButton.addEventListener("click", () => {
    t -= 1;
    if (t < 0) {
      stepButton.disabled = true;
      stepButton.style.backgroundColor = "#7c6f64";
      stepButton.style.cursor = "not-allowed";
      return;
    }
    step(t);
    updatePlot();
  });
});

function plot(xs, ys, t) {
  const canvas = document.getElementById("app");
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#282828";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const radius = 4;
  const padding = 20;

  const minX = Math.min(...xs),
    maxX = Math.max(...xs);
  const minY = Math.min(...ys),
    maxY = Math.max(...ys);

  const xRange = maxX - minX || 1;
  const yRange = maxY - minY || 1;

  const scalePoint = (x, y) => {
    const scaledX =
      padding + ((x - minX) / xRange) * (canvas.width - 2 * padding);
    // y-axis inverted
    const scaledY =
      canvas.height -
      padding -
      ((y - minY) / yRange) * (canvas.height - 2 * padding);
    return [scaledX, scaledY];
  };

  ctx.fillStyle = "#458588";
  for (let i = 0; i < xs.length; i++) {
    const [scaledX, scaledY] = scalePoint(xs[i], ys[i]);
    ctx.beginPath();
    ctx.arc(scaledX, scaledY, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#ebdbb2";
  ctx.font = "12px monospace";
  ctx.fillText(`Timestep: ${NUM_TIMESTEPS - t}`, 10, 15);
}
