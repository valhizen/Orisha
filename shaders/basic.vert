#version 450

// ── Camera matrices ──────────────────────────────────────────────────────────

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos; // xyz = world-space eye position
} camera;

// ── Per-object push constants ────────────────────────────────────────────────

layout(push_constant) uniform PushConstants {
    mat4 model;
    float texBlend;
    float time;
} push;

// ── Vertex inputs (must match Vertex struct in gpu/buffer.rs) ────────────────

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV; // world-space tiled UVs

// ── Outputs to fragment shader ───────────────────────────────────────────────

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPos;
layout(location = 3) out vec2 fragUV;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    gl_Position = camera.proj * camera.view * worldPos;

    // World-space normal — mat3(model) is fine when there is no non-uniform scale.
    fragNormal = normalize(mat3(push.model) * inNormal);
    fragColor = inColor;
    fragWorldPos = worldPos.xyz;
    fragUV = inUV;
}
