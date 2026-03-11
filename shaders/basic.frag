#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;
layout(location = 3) in vec2 fragUV;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos;
} camera;

layout(set = 0, binding = 1) uniform sampler2D diffuseSampler;
layout(set = 0, binding = 2) uniform sampler2D normalSampler;
layout(set = 0, binding = 3) uniform sampler2D specSampler;
layout(set = 0, binding = 4) uniform sampler2D roughSampler;
layout(set = 0, binding = 5) uniform sampler2D dispSampler;

layout(push_constant) uniform PushConstants {
    mat4 model;
    float texBlend;
    float time;
} push;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

mat3 buildTBN(vec3 N) {
    vec3 T_raw = vec3(1.0, 0.0, 0.0) - N * N.x;
    if (dot(T_raw, T_raw) < 1e-4) {
        T_raw = vec3(0.0, 0.0, 1.0) - N * N.z;
    }
    vec3 T = normalize(T_raw);
    vec3 B = cross(T, N);
    return mat3(T, B, N);
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdH = max(dot(N, H), 0.0);
    float denom = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float NdV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

bool isWater(vec3 col) {
    return push.texBlend >= 1.5;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 viewDir = normalize(camera.camPos.xyz - fragWorldPos);
    vec3 lightDir = normalize(vec3(0.6, 1.0, 0.4));
    vec3 lightColor = vec3(1.0, 0.95, 0.85) * 3.0;

    bool water = isWater(fragColor);

    if (water) {
        vec3 waterDeep   = vec3(0.01, 0.06, 0.18);
        vec3 waterShallow = vec3(0.04, 0.20, 0.35);
        float depthFactor = clamp(fragColor.b * 2.0, 0.0, 1.0);
        vec3 waterColor = mix(waterShallow, waterDeep, depthFactor);

        float speed1 = push.time * 0.03;
        float speed2 = push.time * 0.02;
        vec2 wuv1 = fragWorldPos.xz * 0.08 + vec2(speed1, speed1 * 0.7);
        vec2 wuv2 = fragWorldPos.xz * 0.12 + vec2(-speed2 * 0.6, speed2);

        vec3 n1 = texture(normalSampler, wuv1).rgb * 2.0 - 1.0;
        vec3 n2 = texture(normalSampler, wuv2).rgb * 2.0 - 1.0;
        vec3 waterNorm = normalize(n1 + n2);

        mat3 TBN = buildTBN(vec3(0.0, 1.0, 0.0));
        vec3 finalNorm = normalize(TBN * waterNorm);

        float NdL = max(dot(finalNorm, lightDir), 0.0);
        vec3 halfVec = normalize(lightDir + viewDir);
        float spec = pow(max(dot(finalNorm, halfVec), 0.0), 256.0);

        float fresnel = pow(1.0 - max(dot(viewDir, finalNorm), 0.0), 4.0);
        fresnel = clamp(fresnel, 0.1, 0.9);

        vec3 skyReflect = vec3(0.4, 0.55, 0.75);
        vec3 diffuse = waterColor * lightColor * NdL * 0.4;
        vec3 reflection = mix(waterColor, skyReflect, fresnel);
        vec3 specHighlight = lightColor * spec * 1.5;

        vec3 color = reflection * 0.6 + diffuse + specHighlight;
        color += waterColor * 0.08;

        color = color / (color + vec3(1.0));
        color = pow(color, vec3(1.0 / 2.2));
        outColor = vec4(color, 1.0);
        return;
    }

    mat3 TBN = buildTBN(N);
    vec2 uv = fragUV;

    vec3 diffTex  = texture(diffuseSampler, uv).rgb;
    vec3 normRaw  = texture(normalSampler, uv).rgb * 2.0 - 1.0;
    float specTex = texture(specSampler, uv).r;
    float roughTex = texture(roughSampler, uv).r;

    vec3 mappedNorm = normalize(TBN * normalize(normRaw));
    vec3 finalNorm  = normalize(mix(N, mappedNorm, push.texBlend));

    vec3 albedo = mix(fragColor, mix(fragColor, diffTex, 0.45), push.texBlend);

    float roughness = mix(0.9, roughTex, push.texBlend);
    float metallic  = specTex * push.texBlend * 0.04;

    vec3 halfVec = normalize(lightDir + viewDir);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float NDF = distributionGGX(finalNorm, halfVec, roughness);
    float G   = geometrySmith(finalNorm, viewDir, lightDir, roughness);
    vec3  F   = fresnelSchlick(max(dot(halfVec, viewDir), 0.0), F0);

    float NdL = max(dot(finalNorm, lightDir), 0.0);
    float NdV = max(dot(finalNorm, viewDir), 0.001);

    vec3 specular = (NDF * G * F) / (4.0 * NdV * NdL + 0.0001);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 directLight = (kD * albedo / PI + specular) * lightColor * NdL;

    float twoSidedNdL = mix(abs(dot(finalNorm, lightDir)), NdL, push.texBlend);
    directLight = mix(
        albedo * lightColor * twoSidedNdL * 0.30,
        directLight,
        push.texBlend
    );

    vec3 ambient = albedo * 0.10 * vec3(0.85, 0.90, 1.00);

    vec3 color = ambient + directLight;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}
