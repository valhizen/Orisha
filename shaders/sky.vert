#version 450

// ── Sky dome vertex shader ───────────────────────────────────────────────────
//
// The sky dome is always centred on the camera.  We strip the translation from
// the view matrix so the dome never moves.  We output gl_Position.z = gl_Position.w
// so every sky pixel lands at the far plane (depth = 1.0) and is painted behind
// everything else.

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos;
} camera;

layout(push_constant) uniform SkyPush {
    vec4 sunDir;       // xyz = normalised sun direction, w = unused
    float timeOfDay;   // 0..24 hour cycle
    float _pad0;
    float _pad1;
    float _pad2;
} push;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

layout(location = 0) out vec3 fragWorldDir;  // normalised direction from camera

void main() {
    // Strip translation: set column 3 to (0,0,0,1).
    mat4 viewNoTranslation = camera.view;
    viewNoTranslation[3] = vec4(0.0, 0.0, 0.0, 1.0);

    vec4 clipPos = camera.proj * viewNoTranslation * vec4(inPosition, 1.0);

    // Force z = w so the sky always writes depth = 1.0 (the far plane).
    gl_Position = clipPos.xyww;

    // Pass the vertex position as the viewing direction (the dome is centred
    // at the origin so position == direction).
    fragWorldDir = inPosition;
}
