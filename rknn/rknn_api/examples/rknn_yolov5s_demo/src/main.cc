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

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"


using namespace std;
// using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/
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

void parse_det(float * an, const int an_h, const int an_w, const float an_s, vector<vector<float>> an_wh, vector<Det> & res) {
    float c_x, c_y, b_w, b_h, obj_conf, score;
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
                } else {
                    if (obj_conf > 0.1 && val > 0.1) {
                        score = obj_conf * val;
                        Det det;
                        det.c_x = c_x;
                        det.c_y = c_y;
                        det.b_w = b_w;
                        det.b_h = b_h;
                        det.score = score;
                        det.class_id = c % an_vec - 4;
                        res.push_back(det);
                        // std::cout << c_x << "," << c_y << "," << b_w << "," << b_h << "," << score << "," << float(c % an_vec - 4) << "," << std::endl;
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


    // Load RKNN Model
    printf("Loading model ...\n");
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Info
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
    ofstream detfile;
    string detfile_path = "./detfile.txt";
    detfile.open(detfile_path.c_str(), ios::out | ios::trunc );
    // for (string img_name: img_names) {
    for (int img_idx = 0; img_idx < images_cnt; img_idx++) {
        string img_name = img_names[img_idx];
        if (img_name != "StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg") {
            continue;
        }
        std::cout << img_name << std::endl;
        string img_path = val_dir + img_name;
        // Load image
        cv::Mat orig_img = cv::imread(img_path.c_str(), 1);
        cv::Mat img = orig_img.clone();
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", img_path);
            return -1;
        }
        int origin_height = orig_img.rows;
        int origin_width = orig_img.cols;
        int image_height = int(1.0 * image_width / origin_width * origin_height);
        int pad_top = (image_width - image_height) / 2;
        int pad_bottom = image_width - image_height - pad_top;
        cv::resize(orig_img, img, cv::Size(image_width, image_height), (0, 0), (0, 0), cv::INTER_LINEAR);
        cv::copyMakeBorder(img, img, pad_top, pad_bottom, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        printf("input image size is %d, %d\n", img.rows, img.cols);
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

        // Run
        // printf("rknn_run\n");
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

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

        // Post Process
        vector<Det> result;
        parse_det((float *)outputs[2].buf, 20, 20, 32, {{214.0,99.0}, {287.0,176.0}, {376.0,365.0}}, result);
        parse_det((float *)outputs[1].buf, 40, 40, 16, {{94.0,219.0}, {120.0,86.0}, {173.0,337.0}}, result);
        parse_det((float *)outputs[0].buf, 80, 80, 8, {{28.0,31.0}, {53.0,73.0}, {91.0,39.0}}, result);

        for (Det det : result)
        {
            float x0 = int(clip((det.c_x - det.b_w / 2.0) / image_width) * origin_width);
            float y0 = int(clip((det.c_y - det.b_h / 2.0 - pad_top) / image_height) * origin_height);
            float x1 = int(clip((det.c_x + det.b_w / 2.0) / image_width) * origin_width);
            float y1 = int(clip((det.c_y + det.b_h / 2.0 - pad_top) / image_height) * origin_height);
            std::cout << x0 << "," << y0 << "," << x1 << "," << y1 << "," << det.score << "," << det.class_id << std::endl;
            cv::rectangle(orig_img, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255,0,0), 3);
        }

        detfile << "\n";
        cv::imwrite("./result.jpg", orig_img);

        rknn_outputs_release(ctx, 4, outputs);
    };
    detfile.close();

    // Release rknn_outputs
    
    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    std::cout << "done" << std::endl;
    return 0;
}
