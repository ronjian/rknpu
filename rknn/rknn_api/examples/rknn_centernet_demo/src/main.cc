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

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "rknn_api.h"


using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

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
    rknn_context ctx;
    int ret;
    int model_len = 0;
    unsigned char *model;
    const char *model_path = "./centernet_mbv2.rknn";


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
        // if (img_idx > 20) {
        //     break;
        // }
        string img_name = img_names[img_idx];
        // if (img_name.c_str() != "StereoVision_L_990964_10_0_0_5026_D_FakePoop_719_-149.jpeg") {
        //     continue;
        // };
        std::cout << img_name << std::endl;
        string img_path = val_dir + img_name;
        // Load image
        cv::Mat orig_img = cv::imread(img_path.c_str(), 1);
        cv::Mat img = orig_img.clone();
        if(!orig_img.data) {
            printf("cv::imread %s fail!\n", img_path);
            return -1;
        }
        if(orig_img.cols != image_width || orig_img.rows != image_height) {
            // printf("resize %d %d to %d %d\n"
            //     , orig_img.cols, orig_img.rows, image_width, image_height);
            cv::resize(orig_img, img, cv::Size(image_width, image_height), (0, 0), (0, 0), cv::INTER_LINEAR);
        }
        // Set Input Data
        // cv::cvtColor (img, img, cv::COLOR_BGR2RGB);
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
        rknn_output outputs[4];
        memset(outputs, 0, sizeof(outputs));
        outputs[0].want_float = 1;
        outputs[1].want_float = 1;
        outputs[2].want_float = 1;
        outputs[3].want_float = 1;
        ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return -1;
        }
        float * hm = (float *)outputs[0].buf;
        uint32_t hm_size = outputs[0].size;
        float * pool = (float *)outputs[1].buf;
        uint32_t pool_size = outputs[1].size;
        float * wh = (float *)outputs[2].buf;
        uint32_t wh_size = outputs[2].size;
        float * reg = (float *)outputs[3].buf;
        uint32_t reg_size = outputs[3].size;

        // Post Process
        float score;
        int label;
        float width, height, xreg, yreg;
        int x1, y1, x2, y2;
        printf("hm_size %d, pool_size %d, wh_size %d, reg_size %d\n", hm_size, pool_size, wh_size, reg_size);
        ofstream outfile;
        string tgtFile = txt_dir + regex_replace(img_name, regex("jpeg"), "txt");
        string vis_path = vis_dir + img_name;    
        
        outfile.open(tgtFile.c_str(), ios::out | ios::trunc );
        std::cout << tgtFile << std::endl;
        detfile << "/workspace/centernet/data/baiguang/images/val/" + img_name;
        for (int c = 0 ; c < C; c++){
            for (int h = 0; h < H; h++){
                for (int w = 0; w < W; w++){
                    // std::cout << c << ","  << h << "," << w << "," << std::endl;
                    // std::cout << c*H*W + h*W + w << ","  << 0*H*W + h*W + w << "," << 1*H*W + h*W + w << "," << std::endl;
                    if (pool[c*H*W + h*W + w] == hm[c*H*W + h*W + w] && hm[c*H*W + h*W + w] > 0.01) {
                        score = hm[c*H*W + h*W + w];
                        label = c;
                        width = wh[0*H*W + h*W + w];
                        height = wh[1*H*W + h*W + w];
                        xreg = reg[0*H*W + h*W + w];
                        yreg = reg[1*H*W + h*W + w];
                        x1 = (int)(max(((w + xreg) - width / 2.0) * downSampleScale / image_width, 0.0) * origin_width);
                        y1 = (int)(max((((h + yreg) - height / 2.0) * downSampleScale - pad) / image_height, 0.0) * origin_height);
                        x2 = (int)(min(((w + xreg) + width / 2.0) * downSampleScale / image_width, 1.0) * origin_width);
                        y2 = (int)(min((((h + yreg) + height / 2.0) * downSampleScale - pad) / image_height, 1.0) * origin_height);
                        // printf("score is %4.2f, label is %d, width is %4.2f, height is %4.2f, xreg is %4.2f, yreg is %4.2f, x1 is %d, x2 is %d, y1 is %d, y2 is %d\n"
                        //         , score, label, width, height, xreg, yreg, x1, x2, y1, y2);
                        // cv::rectangle(orig_img, Point(x1, y1), Point(x2, y2), Scalar(255, 0, 0, 255), 3);
                        outfile << labelsMap[label] << " " << score << " ";
                        outfile << x1 << " " << y1 << " " << x2 << " " << y2 << "\n";
                        detfile << " " << x1 << "," << y1 << "," << x2 << "," << y2 << "," << label+1 << "," << score;
                    }
                }
            }
        }
        detfile << "\n";
        // cv::imwrite(vis_path, orig_img);
        outfile.close();
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
    return 0;
}
