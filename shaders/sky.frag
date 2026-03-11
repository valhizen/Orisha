#version 450

// ── Procedural atmospheric sky fragment shader ───────────────────────────────
//
// Features:
//   • Rayleigh + Mie scattering (analytical, single-scatter approximation)
//   • Physical sun disc with limb-darkening glow
//   • Full day/night cycle driven by push.timeOfDay (0–24 hours)
//   • Stars visible at night with smooth fade
//   • Horizon haze that intensifies at sunset/sunrise
//   • Smooth twilight transitions (golden hour, blue hour)

layout(location = 0) in vec3 fragWorldDir;

layout(push_constant) uniform SkyPush {
    vec4 sunDir;       // xyz = normalised sun direction, w = unused
    float timeOfDay;   // 0..24 hour cycle
    float _pad0;
    float _pad1;
    float _pad2;
} push;

layout(location = 0) out vec4 outColor;

// ── Constants ────────────────────────────────────────────────────────────────

const float PI = 3.14159265359;

// Rayleigh scattering coefficients (wavelength-dependent: red < green < blue).
const vec3 BETA_R = vec3(5.5e-6, 13.0e-6, 22.4e-6);

// Mie scattering coefficient (wavelength-independent haze).
const float BETA_M = 21.0e-6;

// Mie preferred scattering direction (0 = isotropic, ~0.76 = forward-peaked).
const float MIE_G = 0.76;

// Sun angular radius in radians (~0.53° real, slightly exaggerated for looks).
const float SUN_ANGULAR_RADIUS = 0.0065;

// ── Helper: Rayleigh phase function ──────────────────────────────────────────

float phaseRayleigh(float cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

// ── Helper: Henyey-Greenstein phase function (Mie) ───────────────────────────

float phaseMie(float cosTheta, float g) {
    float g2 = g * g;
    float num = (1.0 - g2);
    float denom = 4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
    return num / denom;
}

// ── Helper: simple pseudo-random hash for star field ─────────────────────────

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float starField(vec3 dir) {
    // Project direction onto a high-res grid for star placement.
    vec3 n = normalize(dir);

    // Only show stars above the horizon.
    if (n.y < 0.0) return 0.0;

    // Spherical coordinates → 2D grid.
    float phi = atan(n.z, n.x);
    float theta = acos(clamp(n.y, -1.0, 1.0));
    vec2 uv = vec2(phi / (2.0 * PI) + 0.5, theta / PI);

    // Tile into a grid; each cell may have one star.
    vec2 gridSize = vec2(400.0, 200.0);
    vec2 cell = floor(uv * gridSize);
    vec2 cellUV = fract(uv * gridSize);

    float starRand = hash(cell);

    // ~3 % of cells get a star.
    if (starRand > 0.03) return 0.0;

    // Star position within cell.
    vec2 starPos = vec2(hash(cell + 1.0), hash(cell + 2.0));
    float dist = length(cellUV - starPos);

    // Tiny bright point with smooth falloff.
    float brightness = smoothstep(0.04, 0.0, dist);

    // Vary star brightness.
    brightness *= 0.5 + 0.5 * hash(cell + 3.0);

    // Twinkle based on a pseudo-time derived from cell hash (slow shimmer).
    float twinkle = 0.7 + 0.3 * sin(hash(cell + 4.0) * 100.0 + push.timeOfDay * 2.5);
    brightness *= twinkle;

    return brightness;
}

// ── Helper: sun altitude factor ──────────────────────────────────────────────
// Returns how "daytime" it is: 1.0 = full day, 0.0 = full night,
// with smooth transitions for dawn/dusk.

float dayFactor(vec3 sunDir) {
    float sunAlt = sunDir.y;  // sine of sun elevation

    // Smooth transition: night when sun is >6° below horizon,
    // full day when sun is >12° above.
    return smoothstep(-0.105, 0.21, sunAlt);
}

// ── Main ─────────────────────────────────────────────────────────────────────

void main() {
    vec3 dir = normalize(fragWorldDir);
    vec3 sunDir = normalize(push.sunDir.xyz);

    // Cosine of angle between view direction and sun.
    float cosTheta = dot(dir, sunDir);

    // ── Atmospheric scattering ───────────────────────────────────────────────
    //
    // Simplified single-scatter model.  We compute optical depth as a function
    // of the view elevation angle: looking straight up has minimal atmosphere,
    // looking at the horizon has maximum path length.

    // Optical depth approximation: more atmosphere near the horizon.
    float viewAlt = max(dir.y, 0.0);
    float opticalDepth = 1.0 / (viewAlt + 0.15);  // avoid division by zero

    // Rayleigh scattering — sky colour.
    vec3 rayleigh = BETA_R * phaseRayleigh(cosTheta) * opticalDepth;

    // Mie scattering — sun/horizon haze.
    float mie = BETA_M * phaseMie(cosTheta, MIE_G) * opticalDepth;

    // ── Day/night colouring ──────────────────────────────────────────────────

    float day = dayFactor(sunDir);
    float sunAlt = sunDir.y;

    // Zenith colour: deep blue by day, dark blue-black at night.
    vec3 zenithDay   = vec3(0.15, 0.35, 0.78);
    vec3 zenithNight  = vec3(0.005, 0.007, 0.02);
    vec3 zenithColour = mix(zenithNight, zenithDay, day);

    // Horizon colour: warm orange-peach during golden hour, cool grey at night.
    vec3 horizonDay    = vec3(0.65, 0.78, 0.92);
    vec3 horizonSunset = vec3(1.0, 0.45, 0.15);
    vec3 horizonNight  = vec3(0.01, 0.012, 0.025);

    // Sunset factor: strongest when sun is near the horizon.
    float sunsetFactor = smoothstep(0.15, 0.0, abs(sunAlt)) * smoothstep(-0.1, 0.05, sunAlt);
    vec3 horizonColour = mix(
        mix(horizonNight, horizonDay, day),
        horizonSunset,
        sunsetFactor
    );

    // Blend between zenith and horizon based on view elevation.
    float horizonBlend = pow(1.0 - max(dir.y, 0.0), 4.0);
    vec3 baseColour = mix(zenithColour, horizonColour, horizonBlend);

    // Apply scattering contribution.
    vec3 scatter = rayleigh * 40.0 + vec3(mie) * 15.0;
    vec3 skyColour = baseColour + scatter * day;

    // ── Sunset/sunrise glow around the sun ───────────────────────────────────

    float sunProximity = max(cosTheta, 0.0);
    float glowIntensity = pow(sunProximity, 8.0) * sunsetFactor * 1.5;
    vec3 glowColour = vec3(1.0, 0.55, 0.2);
    skyColour += glowColour * glowIntensity;

    // ── Sun disc ─────────────────────────────────────────────────────────────

    float sunAngle = acos(clamp(cosTheta, -1.0, 1.0));
    float sunDisc = smoothstep(SUN_ANGULAR_RADIUS, SUN_ANGULAR_RADIUS * 0.7, sunAngle);

    // Sun colour: white-yellow at high altitude, orange-red near horizon.
    vec3 sunColour = mix(vec3(1.0, 0.35, 0.1), vec3(1.0, 0.98, 0.92), smoothstep(-0.05, 0.3, sunAlt));
    float sunBrightness = 30.0 * smoothstep(-0.05, 0.1, sunAlt); // fade sun below horizon

    skyColour += sunColour * sunDisc * sunBrightness;

    // Sun corona / bloom.
    float corona = pow(sunProximity, 256.0) * 8.0 * smoothstep(-0.02, 0.1, sunAlt);
    skyColour += sunColour * corona;

    // ── Stars ────────────────────────────────────────────────────────────────

    float nightFade = 1.0 - day;
    float stars = starField(dir) * nightFade;
    skyColour += vec3(stars * 0.8, stars * 0.85, stars * 1.0);

    // ── Below-horizon darkening ──────────────────────────────────────────────
    //
    // Vertices below the horizon plane (y < 0) darken toward a ground fog colour
    // so the dome doesn't show bright sky below the terrain.

    float belowFactor = smoothstep(0.0, -0.15, dir.y);
    vec3 groundFog = mix(horizonColour * 0.5, vec3(0.01, 0.01, 0.015), 1.0 - day);
    skyColour = mix(skyColour, groundFog, belowFactor);

    // ── Tone mapping (simple Reinhard) ───────────────────────────────────────

    skyColour = skyColour / (skyColour + vec3(1.0));

    // Slight gamma correction for a more natural look on sRGB monitors.
    skyColour = pow(skyColour, vec3(1.0 / 2.2));

    outColor = vec4(skyColour, 1.0);
}
