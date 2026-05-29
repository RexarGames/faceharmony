import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/+esm";

const fileInput = document.querySelector("#fileInput");
const analyzeBtn = document.querySelector("#analyzeBtn");
const resetBtn = document.querySelector("#resetBtn");
const previewImage = document.querySelector("#previewImage");
const emptyState = document.querySelector("#emptyState");
const modelStatus = document.querySelector("#modelStatus");
const resultGrid = document.querySelector("#result");
const mainScore = document.querySelector("#mainScore");
const scoreTitle = document.querySelector("#scoreTitle");
const scoreText = document.querySelector("#scoreText");
const metricsList = document.querySelector("#metricsList");
const scoreRing = document.querySelector(".score-ring");
const overlayCanvas = document.querySelector("#overlayCanvas");
const imageBox = document.querySelector("#imageBox");

let faceLandmarker = null;
let currentImageReady = false;

const clamp = (value, min = 1, max = 10) => Math.max(min, Math.min(max, value));
const round = (value, digits = 1) => Number(value.toFixed(digits));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const closeScore = (value, target, tolerance) => clamp(10 - Math.abs(value - target) / tolerance * 9);
const centerScore = (value, center, tolerance) => clamp(10 - Math.abs(value - center) / tolerance * 9);

initModel();

async function initModel() {
  try {
    modelStatus.textContent = "Загрузка модели...";
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "CPU"
      },
      runningMode: "IMAGE",
      numFaces: 1,
      minFaceDetectionConfidence: 0.55,
      minFacePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
    });

    modelStatus.textContent = "Модель готова";
    analyzeBtn.disabled = !currentImageReady;
  } catch (error) {
    console.error(error);
    modelStatus.textContent = "Не удалось загрузить модель";
    showMessage("Ошибка загрузки", "Проверь интернет или попробуй открыть сайт через Live Server / GitHub Pages.");
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showMessage("Нужна картинка", "Загрузи фото в формате JPG, PNG или WEBP.");
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.onload = () => {
      currentImageReady = true;
      previewImage.style.display = "block";
      emptyState.style.display = "none";
      analyzeBtn.disabled = !faceLandmarker;
      clearCanvas();
      resultGrid.classList.add("hidden");
    };

    previewImage.src = reader.result;
  };

  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!faceLandmarker || !currentImageReady) return;

  modelStatus.textContent = "Анализ...";
  analyzeBtn.disabled = true;

  try {
    const result = faceLandmarker.detect(previewImage);

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      clearCanvas();
      showMessage("Лицо не найдено", "Попробуй фото, где лицо хорошо видно, без сильного поворота и затемнения.");
      return;
    }

    const landmarks = result.faceLandmarks[0];
    const metrics = buildMetrics(landmarks, previewImage);
    const weightedScore = calculateMainScore(metrics);

    drawLandmarks(landmarks);
    renderResult(weightedScore, metrics);
    modelStatus.textContent = "Анализ готов";
  } catch (error) {
    console.error(error);
    showMessage("Ошибка анализа", "Попробуй другое фото или обнови страницу.");
    modelStatus.textContent = "Ошибка анализа";
  } finally {
    analyzeBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  emptyState.style.display = "grid";
  currentImageReady = false;
  analyzeBtn.disabled = true;
  resultGrid.classList.add("hidden");
  clearCanvas();
  modelStatus.textContent = faceLandmarker ? "Модель готова" : "Загрузка модели...";
});

function buildMetrics(lm, img) {
  const faceHeight = dist(lm[10], lm[152]);
  const cheekWidth = dist(lm[234], lm[454]);
  const jawWidth = dist(lm[172], lm[397]);
  const eyeSpan = dist(lm[33], lm[263]);
  const mouthWidth = dist(lm[61], lm[291]);
  const noseWidth = dist(lm[98], lm[327]);
  const faceCenterX = (lm[10].x + lm[152].x + lm[1].x + lm[4].x) / 4;

  const pairs = [
    [33, 263],
    [133, 362],
    [61, 291],
    [234, 454],
    [172, 397],
    [58, 288],
    [136, 365],
    [93, 323]
  ];

  let symmetryError = 0;
  for (const [left, right] of pairs) {
    const lx = lm[left].x;
    const rx = lm[right].x;
    const ly = lm[left].y;
    const ry = lm[right].y;

    const xMirrorError = Math.abs((lx + rx) / 2 - faceCenterX) / cheekWidth;
    const yError = Math.abs(ly - ry) / faceHeight;
    symmetryError += xMirrorError * 0.72 + yError * 0.28;
  }

  symmetryError /= pairs.length;

  const eyeOpenLeft = dist(lm[159], lm[145]) / dist(lm[33], lm[133]);
  const eyeOpenRight = dist(lm[386], lm[374]) / dist(lm[362], lm[263]);
  const eyeOpen = (eyeOpenLeft + eyeOpenRight) / 2;
  const eyeDistanceRatio = eyeSpan / cheekWidth;

  const noseCenter = lm[1].x;
  const mouthCenter = (lm[61].x + lm[291].x) / 2;
  const noseRatio = noseWidth / cheekWidth;
  const mouthRatio = mouthWidth / cheekWidth;
  const faceRatio = faceHeight / cheekWidth;
  const cheekToJaw = cheekWidth / jawWidth;
  const jawRatio = jawWidth / cheekWidth;

  const symmetryScore = clamp(10 - symmetryError * 95);
  const proportionScore = closeScore(faceRatio, 1.42, 0.42);
  const cheekScore = closeScore(cheekToJaw, 1.09, 0.22);
  const jawScore = closeScore(jawRatio, 0.89, 0.20);
  const eyesScore = clamp(
    closeScore(eyeDistanceRatio, 0.61, 0.24) * 0.6 +
    closeScore(eyeOpen, 0.24, 0.20) * 0.4
  );
  const noseScore = clamp(
    closeScore(noseRatio, 0.25, 0.16) * 0.55 +
    centerScore(noseCenter, faceCenterX, cheekWidth * 0.18) * 0.45
  );
  const lipsScore = clamp(
    closeScore(mouthRatio, 0.38, 0.22) * 0.55 +
    centerScore(mouthCenter, faceCenterX, cheekWidth * 0.20) * 0.45
  );

  const photoScore = getPhotoQualityScore(img);

  return [
    {
      key: "symmetry",
      label: "Симметрия",
      score: symmetryScore,
      weight: 0.22,
      text: makeText(symmetryScore, "Левая и правая стороны лица близко совпадают по ключевым точкам.", "Есть заметная разница между сторонами лица или фото снято под углом.")
    },
    {
      key: "proportions",
      label: "Пропорции лица",
      score: proportionScore,
      weight: 0.15,
      text: makeText(proportionScore, "Высота и ширина лица выглядят сбалансированно.", "Пропорция могла просесть из-за ракурса, близкой камеры или наклона головы.")
    },
    {
      key: "cheeks",
      label: "Скулы",
      score: cheekScore,
      weight: 0.14,
      text: makeText(cheekScore, "Ширина в области скул хорошо читается относительно нижней части лица.", "Скулы выражены слабее на этом фото или их скрывает свет/ракурс.")
    },
    {
      key: "jaw",
      label: "Линия челюсти",
      score: jawScore,
      weight: 0.12,
      text: makeText(jawScore, "Нижняя часть лица выглядит достаточно ровной и читаемой.", "Челюсть может выглядеть менее чётко из-за тени, поворота или мягкого освещения.")
    },
    {
      key: "eyes",
      label: "Глаза",
      score: eyesScore,
      weight: 0.12,
      text: makeText(eyesScore, "Положение и открытость глаз хорошо распознаются.", "Оценку могли снизить прищур, очки, тень или поворот головы.")
    },
    {
      key: "nose",
      label: "Нос",
      score: noseScore,
      weight: 0.10,
      text: makeText(noseScore, "Нос расположен близко к центральной оси лица.", "Центр носа или его ширина на фото отклоняются от расчётного баланса.")
    },
    {
      key: "lips",
      label: "Губы",
      score: lipsScore,
      weight: 0.08,
      text: makeText(lipsScore, "Губы ровно расположены относительно центра лица.", "На результат могли повлиять выражение лица, улыбка или поворот головы.")
    },
    {
      key: "photo",
      label: "Качество фото",
      score: photoScore,
      weight: 0.07,
      text: makeText(photoScore, "Фото достаточно светлое, контрастное и резкое для анализа.", "Фото темновато, смазано или слишком плоское по свету.")
    }
  ];
}

function calculateMainScore(metrics) {
  const totalWeight = metrics.reduce((sum, item) => sum + item.weight, 0);
  let score = metrics.reduce((sum, item) => sum + item.score * item.weight, 0) / totalWeight;

  // Небольшое сглаживание, чтобы оценка не была слишком жёсткой из-за одного параметра.
  score = score * 0.92 + 0.55;

  return clamp(round(score));
}

function getPhotoQualityScore(img) {
  const canvas = document.createElement("canvas");
  const size = 128;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, size, size);

  const { data } = ctx.getImageData(0, 0, size, size);
  const lumas = [];

  for (let i = 0; i < data.length; i += 4) {
    const y = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    lumas.push(y);
  }

  const avg = lumas.reduce((a, b) => a + b, 0) / lumas.length;
  const variance = lumas.reduce((sum, y) => sum + (y - avg) ** 2, 0) / lumas.length;
  const contrast = Math.sqrt(variance);

  let edgeSum = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const lap =
        -4 * lumas[i] +
        lumas[i - 1] +
        lumas[i + 1] +
        lumas[i - size] +
        lumas[i + size];
      edgeSum += Math.abs(lap);
    }
  }

  const sharpness = edgeSum / ((size - 2) * (size - 2));

  const brightnessScore = closeScore(avg, 142, 95);
  const contrastScore = closeScore(contrast, 58, 48);
  const sharpnessScore = closeScore(sharpness, 17, 18);

  return clamp(round(brightnessScore * 0.35 + contrastScore * 0.30 + sharpnessScore * 0.35));
}

function makeText(score, good, weak) {
  if (score >= 7.2) return good;
  if (score >= 5.2) return "Параметр выглядит средне: " + weak;
  return weak;
}

function renderResult(score, metrics) {
  resultGrid.classList.remove("hidden");
  mainScore.textContent = score.toFixed(1);
  scoreRing.style.setProperty("--score-progress", score * 10);

  if (score >= 8.1) {
    scoreTitle.textContent = "Очень высокий индекс";
    scoreText.textContent = "Фото набрало сильный результат: хорошая симметрия, читаемые пропорции и нормальное качество изображения.";
  } else if (score >= 6.6) {
    scoreTitle.textContent = "Хороший индекс";
    scoreText.textContent = "Общий результат выше среднего. Часть параметров сильная, часть могла просесть из-за света или ракурса.";
  } else if (score >= 5.0) {
    scoreTitle.textContent = "Средний индекс";
    scoreText.textContent = "Сайт нашёл нормальный баланс, но некоторые параметры могли снизиться из-за качества фото, угла камеры или выражения лица.";
  } else {
    scoreTitle.textContent = "Низкий индекс фото";
    scoreText.textContent = "Скорее всего, фото не очень подходит для анализа: сильный ракурс, тени, размытие или лицо плохо видно.";
  }

  metricsList.innerHTML = "";

  metrics
    .sort((a, b) => b.weight - a.weight)
    .forEach((item) => {
      const metric = document.createElement("div");
      metric.className = "metric";
      metric.innerHTML = `
        <div class="metric-top">
          <span>${item.label}</span>
          <span>${round(item.score)}/10</span>
        </div>
        <div class="bar"><i style="width:${item.score * 10}%"></i></div>
        <p>${item.text}</p>
      `;
      metricsList.appendChild(metric);
    });

  resultGrid.scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawLandmarks(landmarks) {
  const rect = imageBox.getBoundingClientRect();
  const imgRatio = previewImage.naturalWidth / previewImage.naturalHeight;
  const boxRatio = rect.width / rect.height;

  let drawWidth = rect.width;
  let drawHeight = rect.height;
  let offsetX = 0;
  let offsetY = 0;

  if (imgRatio > boxRatio) {
    drawHeight = rect.width / imgRatio;
    offsetY = (rect.height - drawHeight) / 2;
  } else {
    drawWidth = rect.height * imgRatio;
    offsetX = (rect.width - drawWidth) / 2;
  }

  overlayCanvas.width = rect.width * devicePixelRatio;
  overlayCanvas.height = rect.height * devicePixelRatio;
  overlayCanvas.style.width = rect.width + "px";
  overlayCanvas.style.height = rect.height + "px";

  const ctx = overlayCanvas.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const important = [10, 152, 234, 454, 172, 397, 33, 263, 133, 362, 1, 4, 98, 327, 61, 291, 159, 145, 386, 374];

  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(124, 92, 255, 0.65)";
  ctx.fillStyle = "rgba(40, 215, 255, 0.9)";

  for (const index of important) {
    const p = landmarks[index];
    const x = offsetX + p.x * drawWidth;
    const y = offsetY + p.y * drawHeight;

    ctx.beginPath();
    ctx.arc(x, y, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }

  drawLine(ctx, landmarks, offsetX, offsetY, drawWidth, drawHeight, 234, 454);
  drawLine(ctx, landmarks, offsetX, offsetY, drawWidth, drawHeight, 172, 397);
  drawLine(ctx, landmarks, offsetX, offsetY, drawWidth, drawHeight, 33, 263);
  drawLine(ctx, landmarks, offsetX, offsetY, drawWidth, drawHeight, 61, 291);
}

function drawLine(ctx, lm, ox, oy, w, h, a, b) {
  ctx.beginPath();
  ctx.moveTo(ox + lm[a].x * w, oy + lm[a].y * h);
  ctx.lineTo(ox + lm[b].x * w, oy + lm[b].y * h);
  ctx.stroke();
}

function clearCanvas() {
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function showMessage(title, text) {
  resultGrid.classList.remove("hidden");
  mainScore.textContent = "—";
  scoreRing.style.setProperty("--score-progress", 0);
  scoreTitle.textContent = title;
  scoreText.textContent = text;
  metricsList.innerHTML = "";
}
