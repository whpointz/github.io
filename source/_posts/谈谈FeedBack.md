---
title: 谈谈Transform feedback
date: 2017-08-26 22:47:01
tags: [图形]
---

最近在写`OpenGL`的时候，在`Debug`的时候用到了`Transform feedback`，这里简单总结一下。

# Transform feedback
`Transform feedback`是`OpenGL`管线中，顶点处理阶段结束之后，图元装配和光栅化之前的一个步骤。`Transform feedback`可以重新捕获即将装配为图元（点、线段、三角形）的顶点，然后将它们的部分或者全部属性传递到缓存对象中。

实际上，最小的`OpenG`L管线就是一个顶点着色器加上`Transform feedback`的组合--不一定要用到片元着色器。

每当一个顶点传递到图元装配阶段时，将所有需要捕获的属性数据记录到一个或者多个缓存对象中。用户程序可以回读这些缓存对象中的内容，或者`OpenGL`将它们用于后继的渲染工作。

## Debug过程
上面所讲的讲所有需要捕获的属性数据记录到缓存对象中，回读这些缓存对象中的内容也就是Debug的过程。

# Transform feedback 对象
`Transform feedback`状态是封装在一个Transform feedback对象中的。
这个状态包括了：
- 用于记录顶点数据的缓存对象
- 用于标识缓存对象的充满程度的计数器
- 用于标识Transform feedback当前是否启用的状态量

## 分配一个Transform feedback 对象的名称
为`Transform feedback`对象生成n个名称，并且将生成的名称记录到数组`ids`中。
``` c++
void glGenTransformFeedbacks(GLsize n,GLuint* ids);
```
## 将Transform feedback 对象绑定到当前环境
将一个名称为`id`的`Transform feedback`对象绑定到目标`target`上，目标的值必须是`GL_TRANSFORM_FEEDBACK`。
```c++
void glBindTransformFeedback(GLenum target,GLuint id);
```
# Transform feedback 缓存
上面所讲的`Transform feedback`对象是用来管理将顶点捕捉到缓存对象的相关**状态**。
这个状态也包括了当前连接到Transform feedback缓存绑定点的缓存对象。
将名称为`buffer`的缓存对象绑定到目标`target`的某个绑定点上，其索引通过index设置，同时绑定到`target`所设置的一般缓存点上。
``` c++
void glBindBufferBase(GLenum target, GLuint index, Gluint buffer);
```
- `target`参数必须设置为`GL_TRANSFORM_FEEDBACK_BUFFER`。
- `index`必须是当前绑定的`transform feedback`对象的缓存绑定点索引。
- `buffer`表示被绑定的缓存对象的名称。

# transform feedback变量
为了设置transform feedback过程中要记录哪些变量，我们可以调用函数：
``` c++
void glTransformFeedbackVaryings(GLuint program, GLsize count, const GLchar** varyings, GLenum bufferMode);
```

- `program` 指定所用的程序通过
- `count` 设置varyings数组中所包含的字符串的数量
- `varyings` 记录transform feedback的信息
- `bufferMode` 设置的是捕获变量的模式(分离式或者交叉式)

在调用完`glTransformFeedbackVaryings`之后就可以直接调用`glLinkProgram()``来重新链接程序。

# 总体1
``` c++
  // feedback object
  GLuint feedback;
  glGenTransformFeedbacks(1, &feedback);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, feedback);
  // feedback buffer
	GLuint feedbackBuf;
	glGenBuffers(1, &feedbackBuf);
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, feedbackBuf);
	glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 3*10*sizeof(float), NULL, GL_DYNAMIC_COPY);
  // bind buffer to object
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, feedbackBuf);
	static const char* const vars[] = { "TexCoords"};
  // set varyings you want
	glTransformFeedbackVaryings(m_pDrawTextureEffect->m_shaderProg, 1, vars, GL_INTERLEAVED_ATTRIBS);
  // link the shader
	glLinkProgram(m_pDrawTextureEffect->m_shaderProg);
```

# transform feedback的启动和停止
在链接着色器之后，如果得到在`feedback buffer`中的数据。
## 启动
可以在渲染代码之前（即Draw之前）调用函数来启动`transform feedback`。
``` c++
glBeginTransformFeedback(GLenum primitiveMode);
```
`primitiveMode`必须是`GL_POINTS`、`GL_LINES`或者`GL_TRIANGLSE`。和之后的绘制命令中的一样。
## 停止
完成了所有的`transform feedback`图元的渲染，就可以重新切换到正常的渲染模式。
``` c++
glEndTransformFeedback(void);
```

# 总体2
``` c++
//Printf the data in shader
glBeginTransformFeedback(GL_TRIANGLES);
RenderQuad();
glEndTransformFeedback();

float data[8];
glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER,0,8*sizeof(float),data);

for(int ii = 0; ii < 8; ii++){
  printf("%f \t ", data[ii]);
}
printf("\n");
```
