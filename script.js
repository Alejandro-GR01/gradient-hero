window.addEventListener("load", function () {
  const canvas = document.getElementById("canvas");
  const gl = canvas.getContext("webgl");

  // ================= CREAR CANVAS 2D PARA EL GRADIENTE =================
  // Crear canvas2 para el gradiente
  const gradientCanvas = document.createElement("canvas");
  const ctx = gradientCanvas.getContext("2d");

  // Tamaño del gradiente (1D en realidad, ya que solo varía en X)
  gradientCanvas.height = 1; // Solo necesita 1 píxel de altura
  gradientCanvas.width = 256; // 256 píxeles de ancho para suavidad

  // Colores para el gradiente (los mismos que mencionaste)
  const colors = [
    "hsl(240deg 100% 10%)", // #000034
    "hsl(230deg 100% 24%)", // #00107B
    "hsl(225deg 85% 32%)", // #0C2D94
    "hsl(215deg 100% 50%)", // #0162FF
    "hsl(214deg 100% 62%)", // #3C87FF
    "hsl(199deg 89% 57%)" , // #30B7F4
  ];

  // Creando linear gradient en gradientCanvas
  const linearGradient = ctx.createLinearGradient(
    0,
    0, // Inicio
    gradientCanvas.width,
    0 // Fin (horizontal)
  );

  // Agregar paradas de color
  for (const [i, color] of colors.entries()) {
    const stop = i / (colors.length - 1);
    linearGradient.addColorStop(stop, color);
  }

  // Dibujar el gradiente
  ctx.fillStyle = linearGradient;
  ctx.fillRect(0, 0, gradientCanvas.width, gradientCanvas.height);

  // Mostrar el canvas del gradiente en la página (opcional)
  document.body.appendChild(gradientCanvas);
  gradientCanvas.classList.add("hidden");

  // ================= CREAR TEXTURA WEBGL DEL GRADIENTE =================
  const gradientTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, gradientTexture);

  // Configurar parámetros de textura
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  // Cargar los datos del canvas 2D a la textura WebGL
  gl.texImage2D(
    gl.TEXTURE_2D, // target
    0, // level
    gl.RGBA, // internal format
    gl.RGBA, // format
    gl.UNSIGNED_BYTE, // type
    gradientCanvas // source
  );

  gl.bindTexture(gl.TEXTURE_2D, null);

  if (!gl || !ctx) {
    alert("WebGL no soportado");
    document.body.removeChild(canvas);
    document.body.removeChild(gradientCanvas);
    console.log('eliminar canvas y mostrar imagen')
    return;
  }

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  window.addEventListener("resize", resize);
  resize();

  /* ================= SHADERS ================= */

  const vertexShaderSource = /*glsl */ `
    attribute vec2 position;
    varying vec2 v_uv;
    
    void main() {
      v_uv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `;

  const fragmentShaderSource = /*glsl */ `
    precision highp float;
    uniform float u_time;
    uniform float u_width;
    uniform float u_height;
    uniform sampler2D u_gradient;  // NUEVO: Textura del gradiente
   
    #define PI 3.141592653589793
    #define BLUR_AMOUNT 155.0

    const float F = 0.23;
    const float L = 0.0008;
    const float S = 0.13;
    const float A = 2.0;

    // ================= FUNCIONES AUXILIARES PARA SIMPLEX NOISE =================
    
    vec3 mod289(vec3 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    
    vec2 mod289(vec2 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    
    vec4 mod289(vec4 x) {
      return x - floor(x * (1.0 / 289.0)) * 289.0;
    }
    
    vec4 permute(vec4 x) {
      return mod289(((x * 34.0) + 1.0) * x);
    }
    
    vec3 permute(vec3 x) {
      return mod289(((x * 34.0) + 1.0) * x);
    }
    
    vec4 taylorInvSqrt(vec4 r) {
      return 1.79284291400159 - 0.85373472095314 * r;
    }
    
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

      vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
      p0 *= norm.x;
      p1 *= norm.y;
      p2 *= norm.z;
      p3 *= norm.w;

      vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
      m = m * m;
      return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
    }

    float simplex_noise(vec2 v) {
      const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
      
      vec2 i = floor(v + dot(v, C.yy));
      vec2 x0 = v - i + dot(i, C.xx);
      
      vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
      vec4 x12 = x0.xyxy + C.xxzz;
      x12.xy -= i1;

      i = mod289(i);
      vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));

      vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
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
      return 130.0 * dot(m, g);
    }

    // ================= FUNCIONES DE ONDAS =================
    
    float wave_noise( float offset) {
      float noise = 0.0;
      float x_pos = gl_FragCoord.x;
      float offset_time = u_time + offset;
      noise += simplex_noise(vec2(x_pos * (L / 1.00) + F * offset_time, offset_time * S * 1.00)) * A * 0.85;
      noise += simplex_noise(vec2(x_pos * (L / 1.30) + F * offset_time, offset_time * S * 1.26)) * A * 1.15;
      noise += simplex_noise(vec2(x_pos * (L / 1.86) + F * offset_time, offset_time * S * 1.09)) * A * 0.60;
      noise += simplex_noise(vec2(x_pos * (L / 3.25) + F * offset_time, offset_time * S * 0.89)) * A * 0.40;
      return noise;
    }
    
    float calc_blur(float offset) {
      const float blur_L = 0.0018;
      const float blur_S = 0.1;
      const float blur_F = 0.0034;
      float time = u_time * (offset ) ; 
      float x = gl_FragCoord.x;
      
      float noise = simplex_noise(vec2(x * blur_L + blur_F * time, time * blur_S));

      float t = (noise + 1.0) / 2.0;
      t = pow(t, 1.3);
      float blur = mix(1.0, BLUR_AMOUNT, t);
      return blur;
    }

    //suavisa la transicion 
    float smoothstep(float t) {
      return t * t * t * (t * (6.0 * t - 15.0) + 10.0);
    }

    float wave_alpha(float base_y, float wave_height, float x_pos, float y_pos) {
      float offset =  base_y / wave_height;
      float wave_y = base_y + wave_noise( offset) * wave_height;
      float dist = wave_y -  y_pos;
      float blur = calc_blur(offset);
      float alpha = clamp(0.5 + dist / blur , 0.0, 1.0);
      alpha = smoothstep(alpha);
      return alpha;
    }

    // Función background_noise CORREGIDA
    float background_noise(float x_pos, float y_pos, float offset ) {
      float time = u_time + offset;

      float noise = 0.5;
      noise += simplex_noise(vec3(x_pos * L * 1.0 + F * 1.0, y_pos * L * 1.00, time * S)) * 0.30;
      noise += simplex_noise(vec3(x_pos * L * 0.6 + F * 0.6, y_pos * L * 0.85, time * S)) * 0.26;
      noise += simplex_noise(vec3(x_pos * L * 0.4 + F * 0.4, y_pos * L * 0.70, time * S)) * 0.22;
      return clamp(noise, 0.0, 1.0);
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
      
      // Calcular alphas con parámetros correctos
      float wave1_alpha = wave_alpha(WAVE1_Y, WAVE1_HEIGHT, x, y);
      float wave2_alpha = wave_alpha(WAVE2_Y, WAVE2_HEIGHT, x, y);
      float wave3_alpha = wave_alpha(WAVE3_Y, WAVE3_HEIGHT, x, y);
      
      // Calcular el noise de fondo
      float bg_lightness = background_noise(x, y, 0.0);
      float w1_lightness = background_noise(x, y, 200.0);
      float w2_lightness = background_noise(x, y, 400.0);
      float w3_lightness = background_noise(x, y, 600.0);

      float lightness = bg_lightness;
      lightness = mix (lightness,  w1_lightness, wave1_alpha);
      lightness = mix (lightness,  w2_lightness, wave2_alpha);
      lightness = mix (lightness,  w3_lightness, wave3_alpha);
      
      // USAR LA TEXTURA DEL GRADIENTE
      // lightness está entre 0.0 y 1.0, lo usamos como coordenada X
      // La coordenada Y es 0.5 ya que el gradiente es uniforme verticalmente
      vec4 gradientColor = texture2D(u_gradient, vec2(lightness, 0.5));
      
      gl_FragColor = gradientColor;
    }
  `;

  /* ================= WEBGL SETUP ================= */

  function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Error compilando shader:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Error linkando programa:", gl.getProgramInfoLog(program));
  }

  gl.useProgram(program);

  // Configurar buffer de posiciones
  const positions = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

  const positionAttributeLocation = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(positionAttributeLocation);
  gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

  // Obtener locations de todos los uniforms
  const timeUniformLocation = gl.getUniformLocation(program, "u_time");
  const widthUniformLocation = gl.getUniformLocation(program, "u_width");
  const heightUniformLocation = gl.getUniformLocation(program, "u_height");
  const gradientUniformLocation = gl.getUniformLocation(program, "u_gradient");

  // Configurar la textura del gradiente
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, gradientTexture);
  gl.uniform1i(gradientUniformLocation, 0);

  let time = 0;

  function render() {
    time += 0.01;

    // Pasar los valores actuales como uniforms
    gl.uniform1f(timeUniformLocation, time);
    gl.uniform1f(widthUniformLocation, canvas.width);
    gl.uniform1f(heightUniformLocation, canvas.height);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    requestAnimationFrame(render);
  }

  render();
});
