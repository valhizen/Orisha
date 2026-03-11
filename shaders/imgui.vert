#version 450

// ── ImGui vertex shader ──────────────────────────────────────────────────────
//
// Dear ImGui produces 2D screen-space vertices with RGBA8 colours.
// We apply an orthographic projection via push constants to map pixel coords
// to NDC [-1, +1].

layout(push_constant) uniform PushConstants {
    vec2 scale;     // 2.0 / framebuffer_width, 2.0 / framebuffer_height
    vec2 translate; // -1.0, -1.0
} push;

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor; // R8G8B8A8_UNORM → auto-normalised to [0,1]

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec4 fragColor;

void main() {
    fragUV    = inUV;
    fragColor = inColor;
    gl_Position = vec4(inPos * push.scale + push.translate, 0.0, 1.0);
}
