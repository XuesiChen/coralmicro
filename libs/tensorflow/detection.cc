/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "libs/tensorflow/detection.h"

#include <cmath>
#include <queue>

namespace coralmicro::tensorflow {

namespace {
struct ObjectComparator {
  bool operator()(const Object& lhs, const Object& rhs) const {
    return std::tie(lhs.score, lhs.id) > std::tie(rhs.score, rhs.id);
  }
};
}  // namespace

std::string FormatDetectionOutput(const std::vector<Object>& objects) {
  std::string output;
  for (const auto& object : objects) {
    output += "id: " + std::to_string(object.id) +
              " -- score: " + std::to_string(object.score) +
              " -- xmin: " + std::to_string(object.bbox.xmin) +
              " -- ymin: " + std::to_string(object.bbox.ymin) +
              " -- xmax: " + std::to_string(object.bbox.xmax) +
              " -- ymax: " + std::to_string(object.bbox.ymax);
  }
  return output;
}

std::vector<Object> GetDetectionResults(const float* bboxes, const float* ids,
                                        const float* scores, size_t count,
                                        float threshold, size_t top_k) {
  std::priority_queue<Object, std::vector<Object>, ObjectComparator> q;

  for (unsigned int i = 0; i < count; ++i) {
    const int id = std::round(ids[i]);
    const float score = scores[i];
    const float ymin = std::max(0.0f, bboxes[4 * i]);
    const float xmin = std::max(0.0f, bboxes[4 * i + 1]);
    const float ymax = std::max(0.0f, bboxes[4 * i + 2]);
    const float xmax = std::max(0.0f, bboxes[4 * i + 3]);
    if (score < threshold) {
      continue;
    }
    q.push(Object{id, score, BBox<float>{ymin, xmin, ymax, xmax}});
    if (q.size() > top_k) {
      q.pop();
    }
  }

  std::vector<Object> ret;
  ret.reserve(q.size());
  while (!q.empty()) {
    ret.push_back(q.top());
    q.pop();
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

struct YoloDet {
  float x1, y1, x2, y2;
  float conf;
  int cls;
};

static inline float IoU(const YoloDet& a, const YoloDet& b) {
  float xx1 = a.x1 > b.x1 ? a.x1 : b.x1;
  float yy1 = a.y1 > b.y1 ? a.y1 : b.y1;
  float xx2 = a.x2 < b.x2 ? a.x2 : b.x2;
  float yy2 = a.y2 < b.y2 ? a.y2 : b.y2;
  float w = xx2 - xx1, h = yy2 - yy1;
  if (w <= 0.f || h <= 0.f) return 0.f;
  float inter = w * h;
  float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

static void Nms(std::vector<YoloDet>& dets, float iou_thres, size_t max_det, size_t max_nms = 30000) {
  // 1) sort by confidence desc
  std::sort(dets.begin(), dets.end(),
            [](const YoloDet& p, const YoloDet& q) { return p.conf > q.conf; });

  // 2) Ultralytics: if too many boxes, keep only top max_nms before running NMS
  if (dets.size() > max_nms) dets.resize(max_nms);

  std::vector<YoloDet> keep;
  keep.reserve(std::min(dets.size(), max_det));
  std::vector<bool> sup(dets.size(), false);

  // 3) greedy NMS
  for (size_t i = 0; i < dets.size(); ++i) {
    if (sup[i]) continue;
    keep.push_back(dets[i]);
    if (keep.size() >= max_det) break;

    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (sup[j]) continue;
      if (IoU(dets[i], dets[j]) > iou_thres) sup[j] = true;
    }
  }

  // 4) return kept
  dets.swap(keep);
}


static void PrintTensorInfoAndFirstValues(const TfLiteTensor* t, int max_vals = 50) {
  printf("Output size mismatch\r\n");
  printf("Type: %d\r\n", t->type);
  printf("Dims: ");
  for (int i = 0; i < t->dims->size; ++i) printf("%d ", t->dims->data[i]);
  printf("\r\n");

  // Only print floats here (you can extend for int8/uint8 if needed)
  if (t->type == kTfLiteFloat32) {
    const float* data = tflite::GetTensorData<float>(t);
    int total = 1;
    for (int i = 0; i < t->dims->size; ++i) total *= t->dims->data[i];
    int n = (total < max_vals) ? total : max_vals;
    for (int i = 0; i < n; ++i) printf("[%d] %f\r\n", i, data[i]);
  } else {
    printf("Tensor is not float32; add printing for this type if needed.\r\n");
  }
}

// static bool IsYoloV8SingleOutput_1x5xN(const TfLiteTensor* t) {
//   return t &&
//          t->type == kTfLiteFloat32 &&
//          t->dims &&
//          t->dims->size == 3 &&
//          t->dims->data[0] == 1 &&
//          t->dims->data[1] == 5 &&
//          t->dims->data[2] > 0;
// }

static bool IsYoloSingleOutput(const TfLiteTensor* t) {
  if (!t || t->type != kTfLiteFloat32 || !t->dims) return false;

  // Accept [1,5,N] or [1,N,5]
  if (t->dims->size == 3 && t->dims->data[0] == 1) {
    if (t->dims->data[1] == 5 && t->dims->data[2] > 0) return true;  // [1,5,N]
    if (t->dims->data[2] == 5 && t->dims->data[1] > 0) return true;  // [1,N,5]
  }
  return false;
}

static void PrintChannelStats_1x5xN(const TfLiteTensor* out) {
  const int N = out->dims->data[2];
  const float* d = tflite::GetTensorData<float>(out);

  for (int c = 0; c < 5; ++c) {
    float mn = 1e9f, mx = -1e9f;
    for (int i = 0; i < N; ++i) {
      float v = d[c * N + i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    printf("YOLO ch%d[min,max]=[%f,%f]\r\n", c, mn, mx);
  }
}

static inline float clampf(float v, float lo, float hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

struct LetterboxParams {
  float gain;   // min(net_h/orig_h, net_w/orig_w)
  float pad_x;  // (net_w - orig_w*gain)/2
  float pad_y;  // (net_h - orig_h*gain)/2
};


static inline float sigmoidf(float x) {
  return 1.f / (1.f + std::exp(-x));
}

static std::vector<Object> DecodeYoloSingleOutput_ToObjects(const TfLiteTensor* out,
                                                            float conf_thres,
                                                            size_t top_k,
                                                            int img_w,
                                                            int img_h,
                                                            float iou_thres = 0.45f) {

  printf("DecodeYoloSingleOutput_ToObjects(conf_thres=%f iou_thres=%f top_k=%lu)\n",
        (double)conf_thres, (double)iou_thres, (unsigned long)top_k);

  // Determine layout
  // layout A: [1,5,N]  -> d[ch*N + i]
  // layout B: [1,N,5]  -> d[i*5 + ch]
  const int dim1 = out->dims->data[1];
  const int dim2 = out->dims->data[2];
  const bool layout_1x5xN = (dim1 == 5);
  const int N = layout_1x5xN ? dim2 : dim1;

  const float* d = tflite::GetTensorData<float>(out);

  auto get = [&](int i, int ch) -> float {
    return layout_1x5xN ? d[ch * N + i] : d[i * 5 + ch];
  };

  struct Top { float conf; int i; };
  Top best[5] = {{-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}};

  for (int i = 0; i < N; ++i) {
    float c = get(i, 4);
    for (int k = 0; k < 5; ++k) {
      if (c > best[k].conf) {
        for (int t = 4; t > k; --t) best[t] = best[t - 1];
        best[k] = {c, i};
        break;
      }
    }
  }

  printf("Top-5 conf candidates:\n");
  for (int k = 0; k < 5; ++k) {
    int i = best[k].i;
    if (i < 0) continue;
    printf("  k=%d i=%d conf=%f a0=%f a1=%f a2=%f a3=%f\n",
          k, i, best[k].conf, get(i,0), get(i,1), get(i,2), get(i,3));
  }


  // Quick stats for debugging
  float conf_min = 1e9f, conf_max = -1e9f;
  float raw_min  = 1e9f, raw_max  = -1e9f;
  int pass_conf = 0;

  std::vector<YoloDet> dets;
  dets.reserve(128);

  for (int i = 0; i < N; ++i) {
    float a0 = get(i, 0);
    float a1 = get(i, 1);
    float a2 = get(i, 2);
    float a3 = get(i, 3);

    // If conf looks like logits (common), apply sigmoid (Ultralytics-style probability)
    // Heuristic: if outside [0,1] by a decent margin, treat as logit.
    float conf_raw = get(i, 4);
    float conf = conf_raw;
    if (conf_raw < -0.01f || conf_raw > 1.01f) {
      conf = sigmoidf(conf_raw);
    }

    raw_min = std::min(raw_min, conf_raw);
    raw_max = std::max(raw_max, conf_raw);

    conf_min = std::min(conf_min, conf);
    conf_max = std::max(conf_max, conf);

    if (conf < conf_thres) continue;
    pass_conf++;

    printf(
      "YOLO decode: layout=%s N=%d "
      "conf_raw[min,max]=[%f,%f] conf_final[min,max]=[%f,%f] "
      "pass_conf=%d kept_preNMS=%d\r\n",
      layout_1x5xN ? "1x5xN" : "1xNx5",
      N,
      raw_min, raw_max,
      conf_min, conf_max,
      pass_conf,
      (int)dets.size()
    );

    // Decide whether (a0..a3) are normalized
    bool norm_like =
        (a0 >= 0.f && a0 <= 1.5f) &&
        (a1 >= 0.f && a1 <= 1.5f) &&
        (a2 >= 0.f && a2 <= 1.5f) &&
        (a3 >= 0.f && a3 <= 1.5f);

    // Two candidate interpretations:
    // 1) XYWH: (cx,cy,w,h)
    // 2) XYXY: (x1,y1,x2,y2)
    // We'll build both and pick the one that yields valid boxes more often.

    // Convert candidate #1: XYWH -> XYXY
    float cx = a0, cy = a1, w = a2, h = a3;
    if (norm_like) { cx *= img_w; cy *= img_h; w *= img_w; h *= img_h; }
    YoloDet det_xywh;
    det_xywh.x1 = cx - 0.5f * w;
    det_xywh.y1 = cy - 0.5f * h;
    det_xywh.x2 = cx + 0.5f * w;
    det_xywh.y2 = cy + 0.5f * h;

    // Convert candidate #2: XYXY directly
    float x1 = a0, y1 = a1, x2 = a2, y2 = a3;
    if (norm_like) { x1 *= img_w; x2 *= img_w; y1 *= img_h; y2 *= img_h; }
    YoloDet det_xyxy{ x1, y1, x2, y2, conf, 0 };

    auto clamp_box = [&](YoloDet& b) {
      if (b.x1 < 0) b.x1 = 0;
      if (b.y1 < 0) b.y1 = 0;
      if (b.x2 > img_w) b.x2 = img_w;
      if (b.y2 > img_h) b.y2 = img_h;
    };

    clamp_box(det_xywh);
    clamp_box(det_xyxy);

    bool ok_xywh = (det_xywh.x2 > det_xywh.x1) && (det_xywh.y2 > det_xywh.y1);
    bool ok_xyxy = (det_xyxy.x2 > det_xyxy.x1) && (det_xyxy.y2 > det_xyxy.y1);

    // Prefer whichever yields a valid box. If both valid, prefer XYWH (Ultralytics typical path).
    YoloDet det;
    if (ok_xywh) det = det_xywh;
    else if (ok_xyxy) det = det_xyxy;
    else continue;

    det.conf = conf;
    det.cls = 0;
    dets.push_back(det);
  }

  // DEBUG PRINT: you NEED this right now
  printf("YOLO decode: layout=%s N=%d conf[min,max]=[%f,%f] pass_conf=%d kept_preNMS=%d\n",
         layout_1x5xN ? "1x5xN" : "1xNx5", N, conf_min, conf_max, pass_conf, (int)dets.size());

  Nms(dets, iou_thres, top_k, /*max_nms=*/30000);

  std::vector<Object> objs;
  objs.reserve(dets.size());
  for (auto& dd : dets) {
    Object o{};
    o.id = dd.cls;
    o.score = dd.conf;
    o.bbox = BBox<float>{dd.y1, dd.x1, dd.y2, dd.x2};
    objs.push_back(o);
  }
  return objs;
}


std::vector<Object> GetDetectionResults(tflite::MicroInterpreter* interpreter,
                                        float threshold,
                                        size_t top_k) {
  const size_t n_out = interpreter->outputs().size();

  // Case A: YOLOv8-style export: single output [1,5,N]
  if (n_out == 1) {
    TfLiteTensor* out0 = interpreter->output(0);
    printf("out0 type=%d scale=%f zero=%ld\n",
       (int)out0->type, (double)out0->params.scale, (long)out0->params.zero_point);


    if (!IsYoloSingleOutput(out0)) {
      // Not the expected YOLO layout; print and bail
      PrintTensorInfoAndFirstValues(out0);
      return {};
    }

    PrintChannelStats_1x5xN(out0);

    // You need to provide the input size you used for preprocessing.
    // If your model input is 640x640, use that. If you letterbox, use the letterboxed size.
    constexpr int kInputW = 320;
    constexpr int kInputH = 320;

    // If you confirmed coords are normalized (very likely), keep coords_normalized=true.
    return DecodeYoloSingleOutput_ToObjects(out0, threshold, top_k, kInputW, kInputH,
                                       /*iou_thres=*/0.45f);

  }

  // Case B: original SSD-style TFLite detection heads with 4 outputs
  if (n_out == 4) {
    const TfLiteTensor* t0 = interpreter->output(0);
    const TfLiteTensor* t1 = interpreter->output(1);
    const TfLiteTensor* t2 = interpreter->output(2);
    const TfLiteTensor* t3 = interpreter->output(3);

    // Preserve your original “swap” logic, but make it clearer.
    const float *bboxes = nullptr, *ids = nullptr, *scores = nullptr, *count = nullptr;

    // Some models put count at output(2) with dims size == 1, others put it at output(3).
    if (t2->dims->size == 1) {
      scores = tflite::GetTensorData<float>(t0);
      bboxes = tflite::GetTensorData<float>(t1);
      count  = tflite::GetTensorData<float>(t2);
      ids    = tflite::GetTensorData<float>(t3);
    } else {
      bboxes = tflite::GetTensorData<float>(t0);
      ids    = tflite::GetTensorData<float>(t1);
      scores = tflite::GetTensorData<float>(t2);
      count  = tflite::GetTensorData<float>(t3);
    }

    return GetDetectionResults(bboxes, ids, scores,
                               static_cast<size_t>(count[0]),
                               threshold, top_k);
  }

  // Case C: unknown output layout -> debug print first output and return empty
  TfLiteTensor* out0 = interpreter->output(0);
  PrintTensorInfoAndFirstValues(out0);
  return {};
}

}

// namespace coralmicro::tensorflow
