const std = @import("std");
const assert = std.debug.assert;

const NUM_TIMESTEPS = 50;
const BATCH_SIZE = 1000;
const EMBEDDING_SIZE = 128;

const layers = 5;
const weights_data = blk: {
    var weights_array: [layers][]const u8 = undefined;
    for (0..layers) |i| {
        weights_array[i] = @embedFile(std.fmt.comptimePrint("./weights/l{}.weight.npy", .{i + 1}));
    }
    break :blk weights_array;
};
const biases_data = blk: {
    var biases_array: [layers][]const u8 = undefined;
    for (0..layers) |i| {
        biases_array[i] = @embedFile(std.fmt.comptimePrint("./weights/l{}.bias.npy", .{i + 1}));
    }
    break :blk biases_array;
};

fn Matrix(comptime T: type) type {
    return struct {
        shape: [2]usize,
        data: []T,
    };
}

// https://numpy.org/doc/2.1/reference/generated/numpy.lib.format.html#format-version-1-0
fn parse_npy(comptime bytes: []const u8) Matrix(f32) {
    @setEvalBranchQuota(10000);
    if (!std.mem.eql(u8, bytes[0..6], "\x93NUMPY")) {
        @compileError("Invalid magic string");
    }

    const major_version = bytes[6];
    const minor_version = bytes[7];
    if (major_version != 1 or minor_version != 0) {
        @compileError("Unsupported version");
    }

    const header_len = std.mem.readInt(u16, bytes[8..10], .little);
    const header_dict_str = bytes[10 .. 10 + header_len];

    const shape_str = "'shape': (";
    const shape_start = std.mem.indexOf(u8, header_dict_str, shape_str) orelse
        @compileError("No shape in header");
    const shape_end = std.mem.indexOf(u8, header_dict_str[shape_start..], ")") orelse
        @compileError("No end of shape");
    const shape_val = header_dict_str[shape_start + shape_str.len .. shape_start + shape_end];
    var shape_buffer: [2]usize = undefined;
    var shape_count: usize = 0;
    var it = std.mem.split(u8, shape_val, ",");
    while (it.next()) |num_str| {
        const trimmed = std.mem.trim(u8, num_str, " ");
        if (trimmed.len == 0) continue;
        if (shape_count >= shape_buffer.len) @compileError("Unsupported number of dimensions");
        shape_buffer[shape_count] = std.fmt.parseInt(usize, trimmed, 10) catch
            @compileError("Invalid shape");
        shape_count += 1;
    }

    const header_total = 10 + header_len;
    const alignment = 64;
    // data starts after header, aligned to 64 bytes
    const data_start = (header_total + alignment - 1) & ~@as(u32, alignment - 1);
    const data_slice = bytes[data_start..];
    const float_count = @divExact(data_slice.len, @sizeOf(f32));
    const data = @as([*]const f32, @ptrCast(@alignCast(data_slice.ptr)))[0..float_count];

    return .{
        .shape = shape_buffer,
        .data = @constCast(data),
    };
}

const weights = blk: {
    var weights_parsed: [layers]Matrix(f32) = undefined;
    for (0..layers) |i| {
        weights_parsed[i] = parse_npy(weights_data[i]);
    }
    break :blk weights_parsed;
};
const biases = blk: {
    var biases_parsed: [layers]Matrix(f32) = undefined;
    for (0..layers) |i| {
        biases_parsed[i] = parse_npy(biases_data[i]);
    }
    break :blk biases_parsed;
};

const mlp_buffer_sizes = blk: {
    var sizes: [layers]usize = undefined;
    for (0..layers) |i| {
        sizes[i] = BATCH_SIZE * weights[i].shape[1];
    }
    break :blk sizes;
};

var mlp_buffer0 = std.mem.zeroes([mlp_buffer_sizes[0]]f32);
var mlp_buffer1 = std.mem.zeroes([mlp_buffer_sizes[1]]f32);
var mlp_buffer2 = std.mem.zeroes([mlp_buffer_sizes[2]]f32);
var mlp_buffer3 = std.mem.zeroes([mlp_buffer_sizes[3]]f32);
var mlp_buffer4 = std.mem.zeroes([mlp_buffer_sizes[4]]f32);

const mlp_activations = [_]Matrix(f32){
    .{ .shape = [2]usize{ BATCH_SIZE, weights[0].shape[0] }, .data = &mlp_buffer0 },
    .{ .shape = [2]usize{ BATCH_SIZE, weights[1].shape[0] }, .data = &mlp_buffer1 },
    .{ .shape = [2]usize{ BATCH_SIZE, weights[2].shape[0] }, .data = &mlp_buffer2 },
    .{ .shape = [2]usize{ BATCH_SIZE, weights[3].shape[0] }, .data = &mlp_buffer3 },
    .{ .shape = [2]usize{ BATCH_SIZE, weights[4].shape[0] }, .data = &mlp_buffer4 },
};

const sample: Matrix(f32) = .{
    .shape = [2]usize{ BATCH_SIZE, 2 },
    .data = &(struct {
        var buf: [BATCH_SIZE * 2]f32 = undefined;
    }).buf,
};

const new_sample: Matrix(f32) = .{
    .shape = [2]usize{ BATCH_SIZE, 2 },
    .data = &(struct {
        var buf: [BATCH_SIZE * 2]f32 = undefined;
    }).buf,
};

const timesteps: Matrix(usize) = .{
    .shape = [2]usize{ BATCH_SIZE, 1 },
    .data = &(struct {
        var buf: [BATCH_SIZE]usize = undefined;
    }).buf,
};

const input: Matrix(f32) = .{
    .shape = [2]usize{ BATCH_SIZE, 3 * EMBEDDING_SIZE },
    .data = &(struct {
        var buf: [BATCH_SIZE * EMBEDDING_SIZE * 3]f32 = undefined;
    }).buf,
};

fn matrix_multiply(m1: Matrix(f32), m2: Matrix(f32), out: Matrix(f32)) void {
    // inner dimensions are equal
    assert(m1.shape[1] == m2.shape[1]);
    // batch size of input and output match
    assert(out.shape[0] == m1.shape[0]);
    // output features match
    assert(out.shape[1] == m2.shape[0]);

    const M = m1.shape[0]; // rows of m1 and out
    const K = m1.shape[1]; // cols of m1, cols of m2
    const N = m2.shape[0]; // rows of m2 and out

    @memset(out.data, 0);

    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                const m1_idx = i * K + k;
                const m2_idx = j * K + k;
                sum += m1.data[m1_idx] * m2.data[m2_idx];
            }
            const out_idx = i * N + j;
            out.data[out_idx] = sum;
        }
    }
}

fn add_bias(out: Matrix(f32), bias: Matrix(f32)) void {
    assert(out.shape[1] == bias.shape[0]);
    const N = out.shape[1];
    for (0..out.shape[0]) |i| {
        for (0..N) |j| {
            out.data[i * N + j] += bias.data[j];
        }
    }
}

fn ReLU(m: Matrix(f32)) void {
    for (0..m.data.len) |i| {
        m.data[i] = @max(m.data[i], 0);
    }
}

fn sinusoidal_embedding(comptime T: type, x: []T, which: usize, scale: f32) void {
    assert(x.len == BATCH_SIZE);
    const half_size = EMBEDDING_SIZE / 2;
    const emb_initial = @log(10000.0) / @as(f32, (half_size - 1));
    for (0..BATCH_SIZE) |i| {
        for (0..EMBEDDING_SIZE) |j| {
            const out_idx = i * 3 * EMBEDDING_SIZE + which * EMBEDDING_SIZE + j;
            const x_i: f32 = switch (T) {
                f32 => x[i],
                usize => @floatFromInt(x[i]),
                else => unreachable,
            };
            if (j < half_size) {
                const emb = @exp(-emb_initial * @as(f32, @floatFromInt(j)));
                input.data[out_idx] = @sin(x_i * scale * emb);
            } else {
                const emb = @exp(-emb_initial * @as(f32, @floatFromInt(j - half_size)));
                input.data[out_idx] = @cos(x_i * scale * emb);
            }
        }
    }
}

fn model_forward(x: Matrix(f32), t: Matrix(usize)) void {
    // extract columns
    var x0: [BATCH_SIZE]f32 = undefined;
    var x1: [BATCH_SIZE]f32 = undefined;
    for (0..BATCH_SIZE) |i| {
        x0[i] = x.data[i * 2]; // first column is x-coordinates
        x1[i] = x.data[i * 2 + 1]; // second column is y-coordinates
    }

    // positional embeddings
    sinusoidal_embedding(f32, &x0, 0, 25.0);
    sinusoidal_embedding(f32, &x1, 1, 25.0);
    sinusoidal_embedding(usize, t.data, 2, 1.0);

    // MLP layers
    matrix_multiply(input, weights[0], mlp_activations[0]);
    add_bias(mlp_activations[0], biases[0]);
    ReLU(mlp_activations[0]);
    matrix_multiply(mlp_activations[0], weights[1], mlp_activations[1]);
    add_bias(mlp_activations[1], biases[1]);
    ReLU(mlp_activations[1]);
    matrix_multiply(mlp_activations[1], weights[2], mlp_activations[2]);
    add_bias(mlp_activations[2], biases[2]);
    ReLU(mlp_activations[2]);
    matrix_multiply(mlp_activations[2], weights[3], mlp_activations[3]);
    add_bias(mlp_activations[3], biases[3]);
    ReLU(mlp_activations[3]);
    matrix_multiply(mlp_activations[3], weights[4], mlp_activations[4]);
    add_bias(mlp_activations[4], biases[4]);
}

var rnd = std.rand.DefaultPrng.init(42);

const BetaSchedule = enum { Linear, Quadratic };

const NoiseScheduler = struct {
    const beta_start: f32 = 0.0001;
    const beta_end = 0.02;
    var betas: [NUM_TIMESTEPS]f32 = undefined;
    var alphas: [NUM_TIMESTEPS]f32 = undefined;
    var alphas_cumprod: [NUM_TIMESTEPS]f32 = undefined;
    var sqrt_inv_alphas_cumprod: [NUM_TIMESTEPS]f32 = undefined;
    var sqrt_inv_alphas_cumprod_minus_one: [NUM_TIMESTEPS]f32 = undefined;
    var posterior_mean_coef1: [NUM_TIMESTEPS]f32 = undefined;
    var posterior_mean_coef2: [NUM_TIMESTEPS]f32 = undefined;

    fn init(schedule: BetaSchedule) void {
        switch (schedule) {
            .Linear => {
                for (0..NUM_TIMESTEPS) |i| {
                    betas[i] = beta_start + @as(f32, @floatFromInt(i)) * (beta_end - beta_start) / NUM_TIMESTEPS;
                }
            },
            .Quadratic => {
                return;
            },
        }

        for (0..NUM_TIMESTEPS) |i| {
            alphas[i] = 1.0 - betas[i];
        }

        alphas_cumprod[0] = alphas[0];
        for (1..NUM_TIMESTEPS) |i| {
            alphas_cumprod[i] = alphas_cumprod[i - 1] * alphas[i];
        }

        for (0..NUM_TIMESTEPS) |i| {
            sqrt_inv_alphas_cumprod[i] = @sqrt(1.0 / alphas_cumprod[i]);
            sqrt_inv_alphas_cumprod_minus_one[i] = @sqrt(1.0 / alphas_cumprod[i] - 1.0);

            const prev_alpha_cumprod = if (i == 0) 1.0 else alphas_cumprod[i - 1];
            posterior_mean_coef1[i] = betas[i] * @sqrt(prev_alpha_cumprod) / (1.0 - alphas_cumprod[i]);
            posterior_mean_coef2[i] = (1.0 - prev_alpha_cumprod) * @sqrt(alphas[i]) / (1.0 - alphas_cumprod[i]);
        }
    }

    fn reconstruct_x0(t: usize, residuals: []const f32) void {
        const s1 = sqrt_inv_alphas_cumprod[t];
        const s2 = sqrt_inv_alphas_cumprod_minus_one[t];

        const M = sample.shape[0];
        const N = sample.shape[1];
        for (0..M) |i| {
            for (0..N) |j| {
                new_sample.data[i + j * M] = s1 * sample.data[i + j * M] - s2 * residuals[i + j * M];
            }
        }
    }

    fn q_posterior(t: usize) void {
        const s1 = posterior_mean_coef1[t];
        const s2 = posterior_mean_coef2[t];

        const M = sample.shape[0];
        const N = sample.shape[1];
        for (0..M) |i| {
            for (0..N) |j| {
                new_sample.data[i + j * M] = s1 * new_sample.data[i + j * M] + s2 * sample.data[i + j * M];
            }
        }
    }

    fn add_variance(t: usize) void {
        if (t == 0) return;

        var variance = betas[t] * (1.0 - alphas_cumprod[t - 1]) / (1.0 - alphas_cumprod[t]);
        variance = @sqrt(@max(variance, 1e-20));

        const M = sample.shape[0];
        const N = sample.shape[1];
        for (0..M) |i| {
            for (0..N) |j| {
                const noise = rnd.random().floatNorm(f32);
                new_sample.data[i + j * M] += (variance * noise);
            }
        }
    }

    fn step(t: usize, residuals: []const f32) void {
        reconstruct_x0(t, residuals);
        q_posterior(t);
        add_variance(t);
    }
};

export fn init() void {
    const sample_size = sample.shape[0] * sample.shape[1];
    for (0..sample_size) |i| {
        sample.data[i] = rnd.random().floatNorm(f32);
    }
    NoiseScheduler.init(.Linear);
}

export fn step(t: usize) void {
    @memset(timesteps.data, t);
    model_forward(sample, timesteps);
    NoiseScheduler.step(t, mlp_activations[4].data);
    @memcpy(sample.data, new_sample.data);
}

fn run() void {
    init();

    var t: usize = NUM_TIMESTEPS;
    while (t > 0) {
        t -= 1;
        step(t);
    }
    // std.debug.print("new_sample = {any}\n", .{new_sample.data});
}

export fn get_sample_ptr() [*]f32 {
    return sample.data.ptr;
}

export fn get_sample_len() usize {
    return sample.data.len;
}

pub fn main() !void {
    run();
}
