WebAssembly.instantiateStreaming(fetch("main.wasm", { cache: "no-store" }), {
    env: {},
}).then(({ instance }) => {
    const memory = new Float32Array(instance.exports.memory.buffer);
    const sample_ptr = instance.exports.get_sample_ptr() / 4;
    const sample_len = instance.exports.get_sample_len();

    const run = instance.exports.run;
    run();

    const sample = memory.subarray(sample_ptr, sample_ptr + sample_len);
    
    const xs = new Float32Array(Math.floor(sample.length / 2));
    const ys = new Float32Array(Math.floor(sample.length / 2));
    
    for (let i = 0; i < sample.length; i += 2) {
        xs[i/2] = sample[i];
        ys[i/2] = sample[i+1];
    }
    
    plot(xs, ys);
});

function plot(xs, ys) {
    const canvas = document.getElementById("app");
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#282828";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const radius = 4;

    const minX = Math.min(...xs),
        maxX = Math.max(...xs);
    const minY = Math.min(...ys),
        maxY = Math.max(...ys);

    const scalePoint = (x, y) => {
        const scaledX = ((x - minX) / (maxX - minX)) * (canvas.width - 2 * radius);
        // y-axis inverted
        const scaledY =
            canvas.height -
            ((y - minY) / (maxY - minY)) * (canvas.height - 2 * radius);
        return [scaledX, scaledY];
    };

    ctx.fillStyle = "#458588";
    for (let i = 0; i < xs.length; i++) {
        const [scaledX, scaledY] = scalePoint(xs[i], ys[i]);
        ctx.beginPath();
        ctx.arc(scaledX + radius, scaledY + radius, radius, 0, Math.PI * 2);
        ctx.fill();
    }
}
