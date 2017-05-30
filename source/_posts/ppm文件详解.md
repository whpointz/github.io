---
title: PPM文件详解
date: 2017-05-22 19:40:19
tags: [图像]
---

# 前言
最近~~参考~~抄袭大牛的[光线追踪算法](http://www.kevinbeason.com/smallpt/)的时候用到了PPM文件，这里简单的介绍一下PPM文件。

# netpbm介绍
PPM（Portable PixMap）文件是一种图像文件，是[netpbm项目](https://en.wikipedia.org/wiki/Netpbm)定义的一系列Portable图像格式中的像素图像。

netpbm图像格式都比较直接，和平台无关，所以被称为portable（便携式）。所以Portable图像是没有压缩的，图像相对与别的存储格式相比较大，但是因为其简单的格式，一般作为图像处理的中间文件，或者作为简单的图像格式保存。
<!-- more -->
# netpbm格式介绍
netpbm的几种图片格式是通过其表示的颜色类型来区别的。
- PBM（Portable BitMap）是位图
- PGM（Portable GrayMap）是灰度图
- PPM（Portable PixMap）是像素图

## 文件头
- 第一部分 magic number：每一个netpbm图像的前两个字节（ASCII格式）就是magic number，用来标识文件的类型和编码：

| magic number  | 类型   | 编码  |
| ------------ | ------------ | ------------ |
| P1  | PBM 位图  | ASCII  |
| P2  | PGM 灰度图  |  ASCII |
| P3  | PPM 像素图   | ASCII  |
| P4  | PBM 位图    | binary  |
| P5  | PGM 灰图    | binary  |
| P6  | PPM 像素图    | binary  |

> 编码（ASCII和binary）
>
> ASCII编码的文件对于阅读是友好的，可以使用文本编辑器打开，数据和数据之间会有空格和回车隔开。
>
> binary编码则是以二进制的形式顺序存储图像信息，没有空格和回车来分割。
>
> binary格式的文件处理起来更快，图片会更小。ASCII格式的文件阅读和调试起来会更方便。

- 第二部分 图像的列和行，以ASCII编码表示
- 第三部分 描述像素的某个通道的最大颜色值，介于1和65535之间，每个通道最多可以使用两个字节（即16个bit所能表示的最大值，2^17-1 = 65535）表示。

## 图像数据部分
这里只介绍PPM文件，即P3或者P6。
### P3 ASCII
对于P3，其图像的像素的每个通道都由ASCII码组成。比如一个通道的值为255，在ASCII码中就是32 35 35（3个字节）。

使用`fprintf`写入文件，详见代码。

### P6 binary
对于P6（最大颜色值为255时），其图像的像素的每个通道为1个字节。

使用fwrite写入文件中，详见代码。

> `fwrite(const void* buffer, size_t size, size_t count, FILE* file);`
>
> `buffer`是要写入数据的地址，`size`是块长度，`count`是块数量，实际读取长度为`size*count`，返回值为成功读取的块的数量，一般为`count`。

# 写入文件的代码

``` C++
#include <stdio.h>
#include <stdlib.h>

#define width 8
#define height 2
unsigned char img[width*height*3]={
  255,0,0, 0,255,0, 0,0,255, 255,0,0, 0,255,0, 0,0,255, 255,0,0, 0,255,0,
  255,0,0, 0,255,0, 0,0,255, 255,0,0, 0,255,0, 0,0,255, 255,0,0, 0,255,0};

void writePPMDataP6(FILE* f,unsigned char* img){
  fwrite(img,width*height,3,f);
}

void writePPMDataP3(FILE* f,unsigned char* img){
  //注意先height，然后width
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      fprintf(f,"%d %d %d ",img[i*width*3+3*j],img[i*width*3+3*j+1],img[i*width*3+3*j+2]);
    }
    fprintf(f,"\n");
  }
}

int main (){
    FILE* f6 = fopen("testP6.ppm","w");
    fprintf(f6,"P6 ");
    fprintf(f6,"%d %d ",width,height);
    fprintf(f6,"255 ");
    writePPMDataP6(f6,img);
    fclose(f6);

    FILE* f3 = fopen("testP3.ppm","w");
    fprintf(f3,"P3 ");
    fprintf(f3,"%d %d ",width,height);
    fprintf(f3,"255 ");
    writePPMDataP3(f3,img);
    fclose(f3);
}

```

# 结果分析

## 在通道最大值为255的情况下

以上可视化的图像为：

![mark](http://o9z9uibed.bkt.clouddn.com/image/20170530/123858882.png?imageslim)

P3的PPM文件的二进制形式，其像素的每个通道是由ASCII码来表示：

![mark](http://o9z9uibed.bkt.clouddn.com/image/20170530/214716908.png?imageslim)

P6的PPM文件的二进制形式，其像素的每个通道都是一个字节来表示：

![mark](http://o9z9uibed.bkt.clouddn.com/image/20170530/214145216.png?imageslim)

## 对于通道最大值不为255的情况下，使用1个字节作为通道的值
当通道小于255的时候，每个通道都是unsigned char形式，所以会自动溢出取整。比如通道最大值为128(表示纯色的RGB值),当前通道的值为255-->会取成127。

## 当通道大于255的时候，使用2个字节作为通道的值
比如当通道的最大值为260的时候，原来的(255 0 0   0 255 0 )表示的是两个像素6个通道就变成了一个像素3个通道。

而且这两个通道的值需要%256。

# 后续
对通道的最大值的分析是基于我修改了参数，然后看效果的基础上，所以有可能不太对。

一般情况下，通道的最大值都为255，如果有特殊的需求，请阅读源码。
