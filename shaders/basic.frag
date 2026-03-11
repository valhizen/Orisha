#version 450

// ── Fragment inputs (interpolated from vertex outputs) ───────────────────────

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

// ── Output to framebuffer ────────────────────────────────────────────────────

layout(location = 0) out vec4 outColor;

void main() {
    // Simple directional light from upper-right
    vec3 lightDir = normalize(vec3(1.0, 1.0, 0.5));
    vec3 normal = normalize(fragNormal);

    // Two-sided lighting: handles models with inverted normals on mirrored geometry.
    float diffuse = abs(dot(normal, lightDir));
    float ambient = 0.15;

    vec3 color = fragColor * (ambient + diffuse);
    outColor = vec4(color, 1.0);
}
