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
// #include "timer.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"

using namespace std;
using namespace cv;

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


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const int img_width = 224;
    const int img_height = 224;
    const int img_channels = 3;

    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;

    const char *model_path = argv[1];
    const char *img_path = argv[2];

    // Load image
    auto tic = getTimeInUs();
    cv::Mat orig_img = cv::imread(img_path, 1);
    cv::Mat img = orig_img.clone();
    if(!orig_img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }
    if(orig_img.cols != img_width || orig_img.rows != img_height) {
        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, img_width, img_height);
        cv::resize(orig_img, img, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    printf("Load image costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 7.653ms

    // Load RKNN Model
    tic = getTimeInUs();
    model = load_model(model_path, &model_len);
    ret = rknn_init(&ctx, model, model_len, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    printf("Load model costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 55754.215ms

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
    printf("Prepare costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);

    const int loop_count = 100;
    float inference_cost = 0.0;
    float postprecess_cost = 0.0;
    for (int i = 0; i < loop_count; i++){
        // Set Input Data
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
        printf("rknn_run\n");
        tic = getTimeInUs();
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }

        // Get Output
        rknn_output outputs[1];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        ret = rknn_outputs_get(ctx, 1, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        // printf("Inference costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
        inference_cost += (getTimeInUs() - tic) / 1000.0f;

        // Post Process
        tic = getTimeInUs();
        for (int i = 0; i < output_attrs[0].n_elems; i++) {
            float val = ((float*)(outputs[0].buf))[i];
            if (val > 0.01) {
                printf("%d - %f\n", i, val);
            }
        }

        // Release rknn_outputs
        rknn_outputs_release(ctx, 1, outputs);
        // printf("PostPrecess costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
        postprecess_cost += (getTimeInUs() - tic) / 1000.0f;
    }
    printf("Inference costs %8.3fms\n", inference_cost / loop_count); // 10.172ms
    printf("PostPrecess costs %8.3fms\n", postprecess_cost / loop_count); // 0.380ms

    // Release
    if(ctx >= 0) {
        rknn_destroy(ctx);
    }
    if(model) {
        free(model);
    }
    return 0;
}
