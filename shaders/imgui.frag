#version 450

// ── ImGui fragment shader ────────────────────────────────────────────────────
//
// Samples the font atlas and multiplies by the per-vertex colour (with alpha).

layout(set = 0, binding = 0) uniform sampler2D fontAtlas;

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = fragColor * texture(fontAtlas, fragUV);
}
