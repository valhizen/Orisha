#version 450

// ── Camera matrices (updated once per frame via uniform buffer) ──────────────

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} camera;

// ── Per-object transform (pushed inline with draw commands — fast, no alloc) ─

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

// ── Vertex inputs (must match Vertex struct in gpu/buffer.rs) ────────────────

layout(location = 0) in vec3 inPosition;   // Object-space position
layout(location = 1) in vec3 inNormal;     // Object-space normal
layout(location = 2) in vec3 inColor;      // Per-vertex RGB color

// ── Outputs to fragment shader (interpolated across the triangle) ────────────

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;

void main() {
    // Transform vertex to world space, then to clip space
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;

    // Pass world-space normal (assumes no non-uniform scaling, so mat3 is fine)
    fragNormal = mat3(push.model) * inNormal;
    fragColor = inColor;
    fragWorldPos = worldPos.xyz;
}
