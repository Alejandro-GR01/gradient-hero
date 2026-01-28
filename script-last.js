window.addEventListener("load", function () {
  // ================= CANVAS / GL =================
  const canvas = document.createElement("canvas");
  canvas.id = "canvas";
  const gl = canvas.getContext("webgl");

  if (!gl) {
    console.log("WebGL no soportado");
    return;
  }

  // ================= CHEQUEO DE CAPACIDAD DE GPU =================
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  const maxRenderbufferSize = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE);
  const maxVertexTextureUnits = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);
  const maxFragmentUniformVectors = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);

  const MIN_TEXTURE_SIZE = 2048;
  const MIN_RENDERBUFFER = 2048;
  const MIN_VERTEX_TEXTURE_UNITS = 4;
  const MIN_FRAGMENT_UNIFORMS = 32;

  if (
    maxTextureSize < MIN_TEXTURE_SIZE ||
    maxRenderbufferSize < MIN_RENDERBUFFER ||
    maxVertexTextureUnits < MIN_VERTEX_TEXTURE_UNITS ||
    maxFragmentUniformVectors < MIN_FRAGMENT_UNIFORMS
  ) {
    console.log("GPU no cumple los requisitos mínimos");
    return;
  }

  // ================= HERO Y APPEND =================
  const heroSection = document.querySelector(".hero-section");
  if (!heroSection) {
    console.log("No se encontró la sección hero");
    return;
  }
  heroSection.appendChild(canvas);

  // ================= GRADIENT CANVAS & DRAW FUNC =================
  const gradientCanvas = document.createElement("canvas");
  const ctx = gradientCanvas.getContext("2d");

  // limitar ancho del gradiente para ahorrar trabajo (ajustable)
  let gradientWidth = Math.min(512, window.innerWidth);
  gradientCanvas.height = 1;
  gradientCanvas.width = gradientWidth;

  const colors = [
    "hsl(240deg 100% 10%)",
    "hsl(230deg 100% 24%)",
    "hsl(225deg 85% 32%)",
    "hsl(215deg 100% 50%)",
    "hsl(214deg 100% 62%)",
    "hsl(199deg 89% 57%)",
  ];

  function drawGradient(width) {
    gradientCanvas.width = width;
    const linearGradient = ctx.createLinearGradient(0, 0, width, 0);
    colors.forEach((c, i) => linearGradient.addColorStop(i / (colors.length - 1), c));
    ctx.fillStyle = linearGradient;
    ctx.fillRect(0, 0, width, 1);
  }

  drawGradient(gradientWidth);
  gradientCanvas.classList.add("hidden");
  document.body.appendChild(gradientCanvas);

  // ================= CREAR TEXTURA DEL GRADIENTE =================
  const gradientTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, gradientTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, gradientCanvas);
  gl.bindTexture(gl.TEXTURE_2D, null);

  // ================= SHADERS (tu shader complejo) =================
  const vertexShaderSource = `
    attribute vec2 position;
    varying vec2 v_uv;

    void main() {
      v_uv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentShaderSource =/*glsl*/  `
    precision highp float;

    uniform float u_time;
    uniform float u_width;
    uniform float u_height;
    uniform sampler2D u_gradient;

    #define PI 3.141592653589793
    #define BLUR_AMOUNT 155.0

    const float F = 0.07;
    const float L = 0.0009;
    const float S = 0.09;
    const float A = 1.3;

    /* === simplex noise functions (as in your original shader) === */
    vec3 mod289(vec3 x){ return x - floor(x*(1.0/289.0))*289.0; }
    vec2 mod289(vec2 x){ return x - floor(x*(1.0/289.0))*289.0; }
    vec4 mod289(vec4 x){ return x - floor(x*(1.0/289.0))*289.0; }
    vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
    vec3 permute(vec3 x){ return mod289(((x*34.0)+1.0)*x); }
    vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }

    float simplex_noise(vec3 v) {
      const vec2  C = vec2(1.0/6.0, 1.0/3.0);
      const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

      vec3 i  = floor(v + dot(v, C.yyy));
      vec3 x0 = v - i + dot(i, C.xxx);
      vec3 g = step(x0.yzx, x0.xyz);
      vec3 l = 1.0 - g;
      vec3 i1 = min(g.xyz, l.zxy);
      vec3 i2 = max(g.xyz, l.zxy);
      vec3 x1 = x0 - i1 + C.xxx;
      vec3 x2 = x0 - i2 + C.yyy;
      vec3 x3 = x0 - D.yyy;
      i = mod289(i);
      vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));
      float n_ = 0.142857142857;
      vec3  ns = n_ * D.wyz - D.xzx;
      vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
      vec4 x_ = floor(j * ns.z);
      vec4 y_ = floor(j - 7.0 * x_);
      vec4 x = x_ * ns.x + ns.yyyy;
      vec4 y = y_ * ns.x + ns.yyyy;
      vec4 h = 1.0 - abs(x) - abs(y);
      vec4 b0 = vec4(x.xy, y.xy);
      vec4 b1 = vec4(x.zw, y.zw);
      vec4 s0 = floor(b0) * 2.0 + 1.0;
      vec4 s1 = floor(b1) * 2.0 + 1.0;
      vec4 sh = -step(h, vec4(0.0));
      vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
      vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
      vec3 p0 = vec3(a0.xy, h.x);
      vec3 p1 = vec3(a0.zw, h.y);
      vec3 p2 = vec3(a1.xy, h.z);
      vec3 p3 = vec3(a1.zw, h.w);
      vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
      p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
      vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
    }

    float simplex_noise(vec2 v) {
      const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
      vec2 i = floor(v + dot(v, C.yy));
      vec2 x0 = v - i + dot(i, C.xx);
      vec2 i1 = (x0.x > x0.y) ? vec2(1.0,0.0) : vec2(0.0,1.0);
      vec4 x12 = x0.xyxy + C.xxzz;
      x12.xy -= i1;
      i = mod289(i);
      vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
      vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
      m = m * m;
      m = m * m;
      vec3 x = 2.0 * fract(p * C.www) - 1.0;
      vec3 h = abs(x) - 0.5;
      vec3 ox = floor(x + 0.5);
      vec3 a0 = x - ox;
      m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
      vec3 g;
      g.x = a0.x * x0.x + h.x * x0.y;
      g.yz = a0.yz * x12.xz + h.yz * x12.yw;
      return 130.0 * dot(m,g);
    }

    // === wave noise / blending (igual que tu original) ===
    float wave_noise(float offset) {
      float noise = 0.0;
      float x_pos = gl_FragCoord.x;
      float offset_time = u_time + offset;
      noise += simplex_noise(vec2(x_pos*(L/1.00)+F*offset_time, offset_time*S*1.00))*A*0.85;
      noise += simplex_noise(vec2(x_pos*(L/1.30)+F*offset_time, offset_time*S*1.26))*A*1.15;
      noise += simplex_noise(vec2(x_pos*(L/1.86)+F*offset_time, offset_time*S*1.09))*A*0.60;
      noise += simplex_noise(vec2(x_pos*(L/3.25)+F*offset_time, offset_time*S*0.89))*A*0.40;
      return noise;
    }

    float calc_blur(float offset) {
      const float blur_L = 0.0018;
      const float blur_S = 0.1;
      const float blur_F = 0.0034;
      float time = u_time * offset;
      float x = gl_FragCoord.x;
      float noise = simplex_noise(vec2(x*blur_L + blur_F*time, time*blur_S));
      float t = (noise+1.0)/2.0;
      t = pow(t,1.3);
      float blur = mix(1.0, BLUR_AMOUNT, t);
      return blur;
    }

    float smoothstep_custom(float t) {
      return t*t*t*(t*(6.0*t-15.0)+10.0);
    }

    float wave_alpha(float base_y, float wave_height, float x_pos, float y_pos) {
      float offset = base_y / wave_height;
      float wave_y = base_y + wave_noise(offset)*wave_height;
      float dist = wave_y - y_pos;
      float blur = calc_blur(offset);
      float alpha = clamp(0.5 + dist/blur, 0.0, 1.0);
      alpha = smoothstep_custom(alpha);
      return alpha;
    }

    float background_noise(float x_pos, float y_pos, float offset) {
      float time = u_time + offset;
      float noise = 0.5;
      noise += simplex_noise(vec3(x_pos*L*1.0 + F*1.0, y_pos*L*1.0, time*S))*0.30;
      noise += simplex_noise(vec3(x_pos*L*0.6 + F*0.6, y_pos*L*0.85, time*S))*0.26;
      noise += simplex_noise(vec3(x_pos*L*0.4 + F*0.4, y_pos*L*0.70, time*S))*0.22;
      return clamp(noise,0.0,1.0);
    }

    void main() {
      float x = gl_FragCoord.x;
      float y = gl_FragCoord.y;

      float WAVE1_HEIGHT = u_height * 0.07;
      float WAVE2_HEIGHT = u_height * 0.08;
      float WAVE3_HEIGHT = u_height * 0.06;
      float WAVE1_Y = 0.8 * u_height;
      float WAVE2_Y = 0.5 * u_height;
      float WAVE3_Y = 0.20 * u_height;

      float wave1_alpha = wave_alpha(WAVE1_Y, WAVE1_HEIGHT, x, y);
      float wave2_alpha = wave_alpha(WAVE2_Y, WAVE2_HEIGHT, x, y);
      float wave3_alpha = wave_alpha(WAVE3_Y, WAVE3_HEIGHT, x, y);

      float bg_lightness = background_noise(x, y, 0.0);
      float w1_lightness = background_noise(x, y, 100.0);
      float w2_lightness = background_noise(x, y, 200.0);
      float w3_lightness = background_noise(x, y, 300.0);

      float lightness = bg_lightness;
      lightness = mix(lightness, w1_lightness, wave1_alpha);
      lightness = mix(lightness, w2_lightness, wave2_alpha);
      lightness = mix(lightness, w3_lightness, wave3_alpha);

      vec4 gradientColor = texture2D(u_gradient, vec2(lightness, 0.5));

      gl_FragColor = gradientColor;
    }
  `;

  // ================= SHADER COMPILATION / PROGRAM =================
  function createShaderCompiled(gl, type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error("Shader compile error:", gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  const vs = createShaderCompiled(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fs = createShaderCompiled(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  if (!vs || !fs) return;

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program link error:", gl.getProgramInfoLog(program));
    return;
  }
  gl.useProgram(program);

  // ================= BUFFERS =================
  const positions = new Float32Array([-1,-1, 1,-1, -1,1, 1,1]);
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

  const positionAttributeLocation = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(positionAttributeLocation);
  gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

  // ================= UNIFORMS =================
  const timeUniformLocation = gl.getUniformLocation(program, "u_time");
  const widthUniformLocation = gl.getUniformLocation(program, "u_width");
  const heightUniformLocation = gl.getUniformLocation(program, "u_height");
  const gradientUniformLocation = gl.getUniformLocation(program, "u_gradient");

  // configurar textura en unidad 0
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, gradientTexture);
  gl.uniform1i(gradientUniformLocation, 0);

  // ================= RENDER LOOP VARIABLES =================
  const FPS = 20;
  const FRAME_TIME = 1000 / FPS;

  let time = 0;
  let lastTime = 0;
  let frameCount = 0;
  let lastFPSCheck = performance.now();

  // ================= updateCanvasSize (dibujar inmediato) =================
  function updateCanvasSize() {
    const rect = heroSection.getBoundingClientRect();

    // solo si cambió tamaño (evita trabajo innecesario)
    if (canvas.width === rect.width && canvas.height === rect.height) return;

    canvas.width = rect.width;
    canvas.height = rect.height;
    gl.viewport(0, 0, canvas.width, canvas.height);

    // si cambió el ancho del gradiente, recomponer y subir la textura
    const newGradW = Math.min(512, window.innerWidth);
    if (newGradW !== gradientWidth) {
      gradientWidth = newGradW;
      drawGradient(gradientWidth);
      gl.bindTexture(gl.TEXTURE_2D, gradientTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, gradientCanvas);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // dibujar un frame inmediato para evitar parpadeo
    gl.uniform1f(timeUniformLocation, time);
    gl.uniform1f(widthUniformLocation, canvas.width);
    gl.uniform1f(heightUniformLocation, canvas.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // opcional: evitar que el render loop piense que acaba de dibujar hace mucho
    lastTime = performance.now();
  }

  window.addEventListener("resize", updateCanvasSize);
  updateCanvasSize(); // llamada inicial

  // ================= RENDER LOOP =================
  function render(currentTime) {
    if (currentTime - lastTime < FRAME_TIME) {
      requestAnimationFrame(render);
      return;
    }
    lastTime = currentTime;

    // incremento de tiempo (aprox. según FPS objetivo)
    time += 0.016;

    gl.uniform1f(timeUniformLocation, time);
    gl.uniform1f(widthUniformLocation, canvas.width);
    gl.uniform1f(heightUniformLocation, canvas.height);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // FPS debug (opcional)
    frameCount++;
    if (currentTime - lastFPSCheck > 1000) {
      // console.log(`FPS: ${frameCount}`);
      frameCount = 0;
      lastFPSCheck = currentTime;
    }

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);

  // ================= LIMPIEZA =================
  window.addEventListener("beforeunload", () => {
    try {
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      gl.deleteBuffer(positionBuffer);
      gl.deleteTexture(gradientTexture);
    } catch (e) {}
  });
});