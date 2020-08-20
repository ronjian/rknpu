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
    const int img_channels = 3;

    rknn_context ctx1;
    rknn_context ctx2;
    int ret;
    int model_len1 = 0;
    int model_len2 = 0;
    unsigned char *model1;
    unsigned char *model2;

    const char *model_path1 = argv[1];
    const char *img_path1 = argv[2];
    char *img_width_char1 = argv[3];
    const int img_width1 = atoi(img_width_char1);
    char *img_height_char1 = argv[4];
    const int img_height1 = atoi(img_height_char1);

    const char *model_path2 = argv[5];
    const char *img_path2 = argv[6];
    char *img_width_char2 = argv[7];
    const int img_width2 = atoi(img_width_char2);
    char *img_height_char2 = argv[8];
    const int img_height2 = atoi(img_height_char2);

    // Load image
    auto tic = getTimeInUs();
    cv::Mat orig_img1 = cv::imread(img_path1, 1);
    cv::Mat img1 = orig_img1.clone();
    if(!orig_img1.data) {
        printf("cv::imread %s fail!\n", img_path1);
        return -1;
    }
    if(orig_img1.cols != img_width1 || orig_img1.rows != img_height1) {
        printf("resize %d %d to %d %d\n", orig_img1.cols, orig_img1.rows, img_width1, img_height1);
        cv::resize(orig_img1, img1, cv::Size(img_width1, img_height1), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    printf("Load image costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 
    /////
    tic = getTimeInUs();
    cv::Mat orig_img2 = cv::imread(img_path2, 1);
    cv::Mat img2 = orig_img2.clone();
    if(!orig_img2.data) {
        printf("cv::imread %s fail!\n", img_path2);
        return -1;
    }
    if(orig_img2.cols != img_width2 || orig_img2.rows != img_height2) {
        printf("resize %d %d to %d %d\n", orig_img2.cols, orig_img2.rows, img_width2, img_height2);
        cv::resize(orig_img2, img2, cv::Size(img_width2, img_height2), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    printf("Load image costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 

    // Load RKNN Model
    tic = getTimeInUs();
    model1 = load_model(model_path1, &model_len1);
    ret = rknn_init(&ctx1, model1, model_len1, 0);
    model2 = load_model(model_path2, &model_len2);
    ret = rknn_init(&ctx2, model2, model_len2, 0);
    if(ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    printf("Load model costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f); // 

    // Get Model1 Input Output Info
    tic = getTimeInUs();
    rknn_input_output_num io_num1;
    ret = rknn_query(ctx1, RKNN_QUERY_IN_OUT_NUM, &io_num1, sizeof(io_num1));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model1 input num: %d, output num: %d\n", io_num1.n_input, io_num1.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs1[io_num1.n_input];
    memset(input_attrs1, 0, sizeof(input_attrs1));
    for (int i = 0; i < io_num1.n_input; i++) {
        input_attrs1[i].index = i;
        ret = rknn_query(ctx1, RKNN_QUERY_INPUT_ATTR, &(input_attrs1[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs1[i]));
    }

    printf("output 1 tensors:\n");
    rknn_tensor_attr output_attrs1[io_num1.n_output];
    memset(output_attrs1, 0, sizeof(output_attrs1));
    for (int i = 0; i < io_num1.n_output; i++) {
        output_attrs1[i].index = i;
        ret = rknn_query(ctx1, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs1[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs1[i]));
    }
    printf("Prepare 1 costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
    printf("input 1 count is %d\n", io_num1.n_input);

/////////////////////////////
    // Get Model Input Output Info
    tic = getTimeInUs();
    rknn_input_output_num io_num2;
    ret = rknn_query(ctx2, RKNN_QUERY_IN_OUT_NUM, &io_num2, sizeof(io_num2));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model2 input num: %d, output num: %d\n", io_num2.n_input, io_num2.n_output);

    printf("input tensors:\n");
    rknn_tensor_attr input_attrs2[io_num2.n_input];
    memset(input_attrs2, 0, sizeof(input_attrs2));
    for (int i = 0; i < io_num2.n_input; i++) {
        input_attrs2[i].index = i;
        ret = rknn_query(ctx2, RKNN_QUERY_INPUT_ATTR, &(input_attrs2[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(input_attrs2[i]));
    }

    printf("output 2 tensors:\n");
    rknn_tensor_attr output_attrs2[io_num2.n_output];
    memset(output_attrs2, 0, sizeof(output_attrs2));
    for (int i = 0; i < io_num2.n_output; i++) {
        output_attrs2[i].index = i;
        ret = rknn_query(ctx2, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs2[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printRKNNTensor(&(output_attrs2[i]));
    }
    printf("Prepare 2 costs %8.3fms\n", (getTimeInUs() - tic) / 1000.0f);
    printf("input 2 count is %d\n", io_num2.n_input);

    //////////////////////
    const int loop_count = 200;
    float setinput_cost = 0.0;
    float inference_cost = 0.0;
    float getoutput_cost = 0.0;
    // while (1) {
    for (int i = 0; i < loop_count; i++){
        // Set Input Data
        tic = getTimeInUs();
        // rknn_input inputs[1];
        rknn_input inputs1[io_num1.n_input];
        memset(inputs1, 0, sizeof(inputs1));
        printf("inputs1 count is %d\n", io_num1.n_input);
        for (uint32_t in_i = 0; in_i < io_num1.n_input; in_i++) {
            inputs1[in_i].index = in_i;
            inputs1[in_i].type = RKNN_TENSOR_UINT8;
            inputs1[in_i].size = img1.cols*img1.rows*img1.channels();
            inputs1[in_i].fmt = RKNN_TENSOR_NHWC;
            inputs1[in_i].buf = img1.data;
        }

        ret = rknn_inputs_set(ctx1, io_num1.n_input, inputs1);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        ///////////
        rknn_input inputs2[io_num2.n_input];
        memset(inputs2, 0, sizeof(inputs2));
        printf("inputs2 count is %d\n", io_num2.n_input);
        for (uint32_t in_i = 0; in_i < io_num2.n_input; in_i++) {
            inputs2[in_i].index = in_i;
            inputs2[in_i].type = RKNN_TENSOR_UINT8;
            inputs2[in_i].size = img2.cols*img2.rows*img2.channels();
            inputs2[in_i].fmt = RKNN_TENSOR_NHWC;
            inputs2[in_i].buf = img2.data;
        }

        ret = rknn_inputs_set(ctx2, io_num2.n_input, inputs2);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            return -1;
        }
        setinput_cost += (getTimeInUs() - tic) / 1000.0f;
        printf("set input data complete\t");

        // Run
        tic = getTimeInUs();
        printf("rknn_run\n");
        ret = rknn_run(ctx1, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }
        ////////////
        ret = rknn_run(ctx2, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return -1;
        }
        inference_cost += (getTimeInUs() - tic) / 1000.0f;
        printf("inference complete\t");

        // Get Output
        tic = getTimeInUs();
        rknn_output outputs1[1];
        memset(outputs1, 0, sizeof(outputs1));
        outputs1[0].want_float = 1;
        ret = rknn_outputs_get(ctx1, 1, outputs1, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        /////////
        rknn_output outputs2[1];
        memset(outputs2, 0, sizeof(outputs2));
        outputs2[0].want_float = 1;
        ret = rknn_outputs_get(ctx2, 1, outputs2, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }

        getoutput_cost += (getTimeInUs() - tic) / 1000.0f;
        // Release rknn_outputs
        rknn_outputs_release(ctx1, 1, outputs1);
        rknn_outputs_release(ctx2, 1, outputs2);
        printf("get output complete\n");
    }
    printf("setinput_cost %8.3fms\n", setinput_cost / loop_count); // 
    printf("inference_cost %8.3fms\n", inference_cost / loop_count); // 
    printf("getoutput_cost %8.3fms\n", getoutput_cost / loop_count); // 

    // Release
    if(ctx1 >= 0) {
        rknn_destroy(ctx1);
    }
    if(model1) {
        free(model1);
    }

    if(ctx2 >= 0) {
        rknn_destroy(ctx2);
    }
    if(model2) {
        free(model2);
    }

    return 0;
}
