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
#include <sys/time.h>
#include <regex>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"

#include "ssd.h"

using namespace std;
using namespace cv;

#define EVAL_FILE "./baiguang.txt"
#define IMG_DIR "./baiguang/"
#define TXT_DIR "./baiguang-txt/"
#define VIS_DIR "./baiguang-vis/"

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

static int imgs_size = 0;
void readLinesV2(const char *fileName, vector<string> & lines)
{
	FILE* file = fopen(fileName, "r");
	char *s;
	int i = 0;
	int n = 0;
	while ((s = readLine(file, s, &n)) != NULL) {
		lines.emplace_back(string(s));
        imgs_size++;
	}
    printf("img_size: %d\n", imgs_size);
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int img_width = 300;
    const int img_height = 300;
    const int img_channels = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    // const char *img_path = argv[2];

    if (argc != 2) {
        printf("Usage:%s model\n", argv[0]);
        return -1;
    }

    // Get sdk, driver info
    rknn_sdk_version sdk_info;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &sdk_info, sizeof(sdk_info));
    printf("API version is %s, Driver version is %s", sdk_info.api_version, sdk_info.drv_version);

    // Load RKNN Model
    auto tic = getTimeInUs();
    printf("Loading model ...\n");
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    printf("Load model costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 36824.742ms -> 80.300ms

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
    printf("Get Model Input Output Info costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);

    uint64_t setdata_cost = 0.0;
    uint64_t run_cost = 0.0;
    uint64_t getoutput_cost = 0.0;
    uint64_t postprecess_cost = 0.0;
    vector<string> img_names;
    string img_name, img_path, result_path, vis_path;
    readLinesV2(EVAL_FILE, img_names);
    // for (int loop_idx = 0; loop_idx < 10; loop_idx++) {
    for (string img_name: img_names) {
        // img_name = img_names[loop_idx];
        // if (img_name != "StereoVision_L_56262113_2_0_0_5908_D_Shoe_-3336_451.jpeg"){
        //     continue;
        // }
        img_path = string(IMG_DIR) + img_name;
        vis_path = string(VIS_DIR) + img_name;        
        result_path = TXT_DIR + img_name;
        result_path = regex_replace(result_path, regex("jpeg"), "txt");
        printf("img_path %s\n", img_path.c_str());
        printf("vis_path %s\n", vis_path.c_str());
        printf("result_path %s\n", result_path.c_str());
        // Load image
        tic = getTimeInUs();
        cv::Mat orig_img = cv::imread(img_path.c_str(), 1);
        cv::Mat img = orig_img.clone();
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", img_path);
            return -1;
        }
        if(orig_img.cols != img_width || orig_img.rows != img_height) {
            printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
            cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
        }

        // BGR2RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        printf("Load image costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 7.653ms

        // Set Input Data
        tic = getTimeInUs();
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
        setdata_cost += (getTimeInUs() - tic);
        
        // Run
        tic = getTimeInUs();
        printf("rknn_run\n");
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }
        run_cost += (getTimeInUs() - tic);

        // Get Output
        tic = getTimeInUs();
        rknn_output outputs[2];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        outputs[1].want_float = 1;
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        getoutput_cost += (getTimeInUs() - tic);

        // Post Process
        tic = getTimeInUs();
        detect_result_group_t detect_result_group;
        // outputs[0] size is 30672
        // outputs[1] size is 697788
        // outputs[0] size is 230040
        // outputs[1] size is 30672
        // printf("outputs[0] size is %d\n", outputs[0].size);
        // printf("outputs[1] size is %d\n", outputs[1].size);
        postProcessSSD((float *)(outputs[1].buf), (float *)(outputs[0].buf), orig_img.cols, orig_img.rows, &detect_result_group);
        // postProcessSSD((float *)(outputs[0].buf), (float *)(outputs[1].buf), orig_img.cols, orig_img.rows, &detect_result_group);
        // Release rknn_outputs
        rknn_outputs_release(ctx, 2, outputs);
        postprecess_cost += (getTimeInUs() - tic);
        // Draw Objects and Save Results
        ofstream res_fp(result_path);
        for (int i = 0; i < detect_result_group.count; i++) {
            detect_result_t *det_result = &(detect_result_group.results[i]);
            printf("%s @ (%d %d %d %d) %f\n",
                    det_result->name,
                    det_result->box.left, det_result->box.top, det_result->box.right, det_result->box.bottom,
                    det_result->prop);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            // cv::rectangle(orig_img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0, 255), 3);
            // cv::putText(orig_img, det_result->name, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
            res_fp << det_result->name << " " << det_result->prop << " ";
            res_fp << x1 << " " << y1 << " " << x2 << " " << y2 << "\n";
        }
        imwrite(vis_path, orig_img);
        res_fp.close();
    }
    printf("Set data costs %8.3fms\n", setdata_cost / 1000.0 / imgs_size); // 3.170ms
    printf("Run costs %8.3fms\n", run_cost / 1000.0 / imgs_size); // 32.737ms
    printf("Get output costs %8.3fms\n", getoutput_cost / 1000.0 / imgs_size); // 10.955ms
    printf("PostPrecess costs %8.3fms\n", postprecess_cost / 1000.0 / imgs_size); // 7.180ms
    
    // float inference costs:
    // Set data costs   18.307ms
    // Run costs 1552.300ms
    // Get output costs   31.856ms
    // PostPrecess costs   10.028ms

    // ssd_resnet50_v1_fpn_coco.rknn
    // Set data costs    8.413ms
    // Run costs  646.495ms
    // Get output costs  250.442ms
    // PostPrecess costs   12.353ms

    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}