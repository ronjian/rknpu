/****************************************************************************
*
*    Copyright (c) 2017 - 2018 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sys/time.h>
#include "help.h"
#include <math.h>
// #include <algorithm>
#include <sys/time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"


using namespace std;
// using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline uint64_t getTimeInUs() {
    uint64_t time;
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    time = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
    return time;
}

template <typename T>
std::vector<size_t>
argsort_descend(const std::vector<T>& v)
{
  std::vector<size_t> indices(v.size());

  size_t n = 0;
  std::generate(indices.begin(), indices.end(), [&n] { return n++; });

  std::sort(indices.begin(), indices.end(),
            [&v](const size_t i, const size_t j) { return v[i] > v[j]; });

  return indices;
}

float
compute_iou(
  const BoxCornerEncoding& lhs, const BoxCornerEncoding& rhs)
{
  float lhs_area = (lhs.ymax - lhs.ymin) * (lhs.xmax - lhs.xmin);
  float rhs_area = (rhs.ymax - rhs.ymin) * (rhs.xmax - rhs.xmin);

  if (lhs_area <= 0.0 || rhs_area <= 0.0) {
    return 0;
  }


  float intersection_ymin = std::max(lhs.ymin, rhs.ymin);
  float intersection_xmin = std::max(lhs.xmin, rhs.xmin);
  float intersection_ymax = std::min(lhs.ymax, rhs.ymax);
  float intersection_xmax = std::min(lhs.xmax, rhs.xmax);

  float dh = std::max(intersection_ymax - intersection_ymin, 0.0f);
  float dw = std::max(intersection_xmax - intersection_xmin, 0.0f);

  float intersection = dh * dw;

  float area_union = lhs_area + rhs_area - intersection;

  return intersection / area_union;
}

vector<size_t>
nms_single_class (
  const PredictBoxes& boxes,
  const PredictScores& scores,
  float confidence_threshold=0.001,
  float iou_threshold=0.5)
{
//   std::cout << "=========" << std::endl;
  vector<size_t> desc_ind = argsort_descend (scores);
  vector<int> suppression;

  int last_elem = -1;
  for (int i = 0; i < desc_ind.size(); i++) {
    size_t idx = desc_ind[i];
    if (scores[idx] >= confidence_threshold) {
      last_elem = i;

      suppression.push_back (i);
    }
    else {
      break;
    }
  }

  vector<size_t> selected;
  for (int i = 0; i <= last_elem; i++) {
    if (suppression[i] == 99999) {
//      cout << "index " << i << " in score index array is already suppressed.\n";
      // ++i;
      continue;
    }

    size_t idx = desc_ind[i]; /* box index i */
    const BoxCornerEncoding& cur_box = boxes[idx];

    selected.emplace_back (idx);

    int j = i + 1;
    while (j <= last_elem)
    {
      size_t jdx = desc_ind[j]; /* box index j */
      const BoxCornerEncoding& box_j = boxes[jdx];

      float iou = compute_iou (cur_box, box_j);
    //   std::cout << "i:" << i << ",j:" << j << ",idx:" << idx << ",jdx:" << jdx << ",iou:" << iou << std::endl;
      assert (iou >= 0.0);
      /*
       * if iou is above threshold, then suppress box_j.
       * otherwise box_j will be the next *new* box.
       */
      if (iou >= iou_threshold) {
        suppression[j] = 99999;
      }

      ++j;
    }

    //i = j;
  }

  return selected;
}

map<int, cv::Scalar> kColorTable = {
    { 1,          cv::Scalar(0  , 255, 0) },
    { 2,     cv::Scalar(255, 0  , 0) },
    { 3,          cv::Scalar(0  , 0  , 255) },
    { 4,   cv::Scalar(255, 0  , 255) },
    { 5,           cv::Scalar(0  , 255, 255) },
    { 6,   cv::Scalar(255, 255, 0) },
    { 7,    cv::Scalar(0  , 127, 0) },
    { 8, cv::Scalar(0  , 0, 127) },
    { 9,        cv::Scalar(127, 0, 0) },
    { 10,         cv::Scalar(127, 0  , 255) },
    { 11, cv::Scalar(0  , 127, 255) },
    { 12,cv::Scalar(100, 50 , 200) },
    { 13,         cv::Scalar(0  , 127, 255) },
    { 14,   cv::Scalar(127, 255, 255) },
    { 15,   cv::Scalar(255, 127, 255) },
    { 16,         cv::Scalar(255, 255, 127) },
    { 17,   cv::Scalar(127, 127, 255) },
    { 18,     cv::Scalar(127, 255, 127) },
    { 19,     cv::Scalar(255, 127, 127) },
    { 20,   cv::Scalar(127, 100, 127) },
    { 21, cv::Scalar(100, 127, 127) },
    { 22, cv::Scalar(127, 127, 100) },
    { 23,         cv::Scalar(100, 127, 50) },
    { 24, cv::Scalar(50 , 200, 127) },
    { 25,         cv::Scalar(255, 100, 127) },
    { 26,          cv::Scalar(100, 255, 127) },
    { 27,     cv::Scalar(255, 50 , 127) },
    { 28,    cv::Scalar(200, 255, 127) },
    { 29,    cv::Scalar(200, 100, 127) },
};

map<string, int> classname2id = {
                            {"background",0},
                            {"wire",1},
                            {"pet feces",2},
                            {"shoe",3},
                            {"bar stool a",4},
                            {"fan",5},
                            {"power strip",6},
                            {"dock(ruby)",7},
                            {"dock(rubys+tanosv)",8},
                            {"bar stool b",9},
                            {"scale",10},
                            {"clothing item",11},
                            {"cleaning robot",12},
                            {"fan b",13},
                            {"door mark a",14},
                            {"door mark b",15},
                            {"wheel",16},
                            {"door mark c",17},
                            {"flat base",18},
                            {"whole fan",19},
                            {"whole fan b",20},
                            {"whole bar stool a",21},
                            {"whole bar stool b",22},
                            {"fake poop a",23},
                            {"dust pan",24},
                            {"folding chair",25},
                            {"laundry basket",26},
                            {"handheld cleaner",27},
                            {"sock",28},
                            {"fake poop b",29},
    };

struct Det {
    float c_x;
    float c_y;
    float b_w;
    float b_h;
    float score;
    int class_id;
};

//TODO
float clip(float val){
    float res;
    if (val > 1.0){
        res = 1.0;
    } else {
        res = val;
    }

    if (val < 0.0) {
        res = 0.0;
    } else {
        res = val;
    }

    return res;
}
    

static void printRKNNTensor(rknn_tensor_attr *attr) {
    printf("index=%d name=%s n_dims=%d dims=[%d %d %d %d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d fl=%d zp=%d scale=%f\n", 
            attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0], 
            attr->n_elems, attr->size, 0, attr->type, attr->qnt_type, attr->fl, attr->zp, attr->scale);
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    if(fp) {
        fclose(fp);
    }
    return model;
}

void parse_det_arch(float * an, const int an_h, const int an_w, const float an_s
            , vector<vector<float>> an_wh, PredictBoxes & pred_boxes, PredictScores * pred_scores
            , int pad_top, int pad_left, int resize_w, int resize_h, int origin_height, int origin_width
            ) {
    float c_x, c_y, b_w, b_h, obj_conf;
    int class_id;
    for (int h = 0; h < an_h; h++) {
        for (int w = 0; w < an_w; w++) {
            for (int c = 0; c < an_c; c++) {
                float val = an[c * an_h * an_w + h * an_w + w];
                if (c % an_vec == 0) {
                    c_x = val;
                    c_x = (c_x * 2.0 - 0.5 + w) * an_s;
                } else if (c % an_vec  == 1) {
                    c_y = val;
                    c_y = (c_y * 2.0 - 0.5 + h) * an_s;
                } else if (c % an_vec == 2) {
                    b_w = val;
                    b_w = pow((b_w * 2.0), 2) * an_wh[c / an_vec][0];
                } else if (c % an_vec == 3) {
                    b_h = val;
                    b_h = pow((b_h * 2.0), 2) * an_wh[c / an_vec][1];
                } else if (c % an_vec == 4) {
                    obj_conf = val;
                    if (obj_conf > THRESHOLD) {
                        BoxCornerEncoding box = {};
                        box.ymin = clip((c_y - b_h / 2.0 - pad_top) / resize_h) * origin_height;
                        box.xmin = clip((c_x - b_w / 2.0 - pad_left) / resize_w) * origin_width;
                        box.ymax = clip((c_y + b_h / 2.0 - pad_top) / resize_h) * origin_height;
                        box.xmax = clip((c_x + b_w / 2.0 - pad_left) / resize_w) * origin_width;
                        pred_boxes.emplace_back (box);
                    }
                } else {
                    if (obj_conf > THRESHOLD) {
                        pred_scores[c % an_vec - 4].emplace_back (obj_conf * val);
                    }
                }
            }
        }
    }
}

void parse_det(float * an, const int an_h, const int an_w, const float an_s
            , vector<vector<float>> an_wh, PredictBoxes & pred_boxes, PredictScores * pred_scores
            , int pad_top, int pad_left, int resize_w, int resize_h, int origin_height, int origin_width
            ) {
    float c_x, c_y, b_w, b_h, obj_conf, val;
    for (int h = 0; h < an_h; h++) {
        for (int w = 0; w < an_w; w++) {
            for (int a = 0; a < 3; a++) {
                obj_conf = an[(a * an_vec + 4) * an_h * an_w + h * an_w + w];
                if (obj_conf > THRESHOLD) {
                    BoxCornerEncoding box = {};
                    c_x = (an[(a * an_vec + 0) * an_h * an_w + h * an_w + w] * 2.0 - 0.5 + w) * an_s;
                    c_y = (an[(a * an_vec + 1) * an_h * an_w + h * an_w + w] * 2.0 - 0.5 + h) * an_s;
                    b_w = pow((an[(a * an_vec + 2) * an_h * an_w + h * an_w + w] * 2.0), 2) * an_wh[a][0];
                    b_h = pow((an[(a * an_vec + 3) * an_h * an_w + h * an_w + w] * 2.0), 2) * an_wh[a][1];
                    box.ymin = clip((c_y - b_h / 2.0 - pad_top) / resize_h) * origin_height;
                    box.xmin = clip((c_x - b_w / 2.0 - pad_left) / resize_w) * origin_width;
                    box.ymax = clip((c_y + b_h / 2.0 - pad_top) / resize_h) * origin_height;
                    box.xmax = clip((c_x + b_w / 2.0 - pad_left) / resize_w) * origin_width;
                    pred_boxes.emplace_back (box);

                    if (cross_class_nms) {
                        float max_score = 0.0;
                        float score;
                        float max_score_clsid;
                        for (int pos = 5; pos < an_vec; pos++){
                            val = an[(a * an_vec + pos) * an_h * an_w + h * an_w + w];
                            score = obj_conf * val;
                            if (score > max_score){
                                max_score = score;
                                max_score_clsid = pos - 4;
                            }
                        }
                        pred_scores[0].emplace_back (max_score);
                        pred_scores[1].emplace_back (max_score_clsid);
                    } else {
                        for (int pos = 5; pos < an_vec; pos++){
                            val = an[(a * an_vec + pos) * an_h * an_w + h * an_w + w];
                            pred_scores[pos - 4].emplace_back (obj_conf * val);
                        }
                    }

                } 

            }
        }
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
    const char *model_path = "./yolov5s.rknn";

    auto tic = getTimeInUs();
    // Load RKNN Model
    printf("Loading model ...\n");
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    printf("Load model costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);

    // Get Model Input Output Info
    tic = getTimeInUs();
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs[i]));
    }
    printf("Prepare input/output costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
    // ofstream detfile;
    // string detfile_path = "./detfile.txt";
    // detfile.open(detfile_path.c_str(), ios::out | ios::trunc );
    // for (string img_name: img_names) {
    int readcost = 0;
    int resizecost = 0;
    int padcost = 0;
    int cvtcost = 0;
    int inputcost = 0;
    int infercost = 0;
    int outputcost = 0;
    int decodecost = 0;
    int nmscost = 0;
    int nmshandling_cnt = 0;
    for (int img_idx = 0; img_idx < images_cnt; img_idx++) {
    // for (int img_idx = 0; img_idx < 27; img_idx++) {
        string img_name = img_names[img_idx];
        // if (img_name != "StereoVision_L_922887_32_0_1_7156.jpeg") {
        // if (img_name != "StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg") {
        // if (img_name != "StereoVision_L_644600_10_0_0_97.jpeg"
        //     && img_name != "StereoVision_L_341078_0_0_1_1046_D_Sock_-1237_1777.jpeg") {
        //     continue;
        // }
        std::cout << img_name << std::endl;
        string img_path = val_dir + img_name;
        // Load image
        tic = getTimeInUs();
        cv::Mat orig_img = cv::imread(img_path.c_str(), 1);
        cv::Mat img = orig_img.clone();
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", img_path);
            return -1;
        }
        int origin_height = orig_img.rows;
        int origin_width = orig_img.cols;
        readcost += (getTimeInUs() - tic);

        tic = getTimeInUs();
        // int image_height = int(1.0 * image_width / origin_width * origin_height);
        // int pad_top = (image_width - image_height) / 2;
        // int pad_bottom = image_width - image_height - pad_top;
        cv::resize(orig_img, img, cv::Size(resize_w, resize_h), (0, 0), (0, 0), cv::INTER_AREA);
        resizecost += (getTimeInUs() - tic);

        tic = getTimeInUs();
        cv::copyMakeBorder(img, img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        padcost += (getTimeInUs() - tic);

        tic = getTimeInUs();
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cvtcost += (getTimeInUs() - tic);

        tic = getTimeInUs();
        // printf("input image size is %d, %d\n", img.rows, img.cols);
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = img.cols*img.rows*img.channels();
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = img.data;

        ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        inputcost += (getTimeInUs() - tic);

        tic = getTimeInUs();
        // Run
        // printf("rknn_run\n");
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }
        infercost += (getTimeInUs() - tic);
        
        tic = getTimeInUs();
        // Get Output
        rknn_output outputs[3];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        outputs[1].want_float = 1;
        outputs[2].want_float = 1;
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        outputcost += (getTimeInUs() - tic);

        tic = getTimeInUs();
        // Post Process
        ofstream outfile;
        string tgtFile = txt_dir + regex_replace(img_name, regex("jpeg"), "txt");
        outfile.open(tgtFile.c_str(), ios::out | ios::trunc );
        PredictBoxes  pred_boxes;
        PredictScores pred_scores[class_cnt+1];
        
        parse_det((float *)outputs[2].buf, 16, 21, 32, {{214.0,99.0}, {287.0,176.0}, {376.0,365.0}},pred_boxes,pred_scores,pad_top, pad_left, resize_w, resize_h, origin_height, origin_width);
        parse_det((float *)outputs[1].buf, 32, 42, 16, {{94.0,219.0}, {120.0,86.0}, {173.0,337.0}},pred_boxes,pred_scores,pad_top, pad_left, resize_w, resize_h, origin_height, origin_width);
        parse_det((float *)outputs[0].buf, 64 ,84, 8, {{28.0,31.0}, {53.0,73.0}, {91.0,39.0}},pred_boxes,pred_scores,pad_top, pad_left, resize_w, resize_h, origin_height, origin_width);
        nmshandling_cnt += pred_boxes.size();
        decodecost += (getTimeInUs() - tic);
        
        tic = getTimeInUs();
        if (cross_class_nms) {
            const vector<size_t>& selected = nms_single_class (pred_boxes, pred_scores[0], score_threshold, nms_iou);
            for (size_t sel: selected)
            {   
                float score = pred_scores[0][sel];
                int clsid = (int)pred_scores[1][sel];
                string classname = originLabelsMap[clsid];
                int x0 = pred_boxes[sel].xmin;
                int y0 = pred_boxes[sel].ymin;
                int x1 = pred_boxes[sel].xmax;
                int y1 = pred_boxes[sel].ymax;
                if (PLOT && score > score_threshold) {
                    cv::rectangle(orig_img, cv::Point(x0, y0), cv::Point(x1, y1), kColorTable[clsid], 3);
                    cv::putText(orig_img, classname + ":" + to_string(score).substr(0, 4), cv::Point(x0, y0-2) ,0,1,cv::Scalar(225, 255, 255), 2);
                }
                outfile << classname << " " << score << " ";
                outfile << x0 << " " << y0 << " " << x1 << " " << y1 << "\n";
            }
        } else {
            for (int i = 1; i < class_cnt+1; i++){
                // printf("%d, %d", pred_boxes.size(), pred_scores[i].size());
                const vector<size_t>& selected = nms_single_class (pred_boxes, pred_scores[i], score_threshold, nms_iou);

                for (size_t sel: selected)
                {
                    string classname = originLabelsMap[i];
                    float score = pred_scores[i][sel];
                    int x0 = pred_boxes[sel].xmin;
                    int y0 = pred_boxes[sel].ymin;
                    int x1 = pred_boxes[sel].xmax;
                    int y1 = pred_boxes[sel].ymax;
                    if (PLOT && score > score_threshold) {
                        cv::rectangle(orig_img, cv::Point(x0, y0), cv::Point(x1, y1), kColorTable[i], 3);
                        cv::putText(orig_img, classname + ":" + to_string(score).substr(0, 4), cv::Point(x0, y0-2) ,0,1,cv::Scalar(225, 255, 255), 2);
                    }
                    outfile << classname << " " << score << " ";
                    outfile << x0 << " " << y0 << " " << x1 << " " << y1 << "\n";
                }
            }
        }

        if (PLOT) {
            cv::imwrite(vis_dir + img_name, orig_img);
        }


        nmscost += (getTimeInUs() - tic);
        outfile.close();
        rknn_outputs_release(ctx, 4, outputs);
    }
    // detfile.close();

    // Release rknn_outputs
    
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    printf("read image costs %8.3fms\n", float(readcost) / 1000.0f/ images_cnt);
    printf("resize image costs %8.3fms\n", float(resizecost) / 1000.0f/ images_cnt);
    printf("padding image costs %8.3fms\n", float(padcost) / 1000.0f/ images_cnt);
    printf("convert channel costs %8.3fms\n", float(cvtcost) / 1000.0f/ images_cnt);
    printf("input costs %8.3fms\n", float(inputcost) / 1000.0f / images_cnt);
    printf("infer costs %8.3fms\n", float(infercost) / 1000.0f / images_cnt);
    printf("output costs %8.3fms\n", float(outputcost) / 1000.0f / images_cnt);
    printf("decode anchors costs %8.3fms\n", float(decodecost) / 1000.0f / images_cnt);
    printf("NMS costs %8.3fms\n", float(nmscost) / 1000.0f / images_cnt);
    printf("average NMS handling count is %8.3f\n", float(nmshandling_cnt) / images_cnt);

    std::cout << "done" << std::endl;


    return 0;
}
