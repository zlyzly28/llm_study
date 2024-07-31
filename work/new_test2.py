import uvicorn
import os
import gradio as gr
# from utils.inference import predict
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from enum import Enum
import os
from openai import OpenAI
from fastapi import FastAPI, HTTPException
# 加载环境变量
load_dotenv()

#初始化FastAPI应用程序
app = FastAPI()

#定义请求和响应模型
class Request(BaseModel):
    prompt : str

class Response(BaseModel):
    response : str

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url=os.getenv("base_url"),
)

async def predict(prompt, history=None):
    # print(prompt, history)
    history_openai_format = []
    if history is not None:
        for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human})
            history_openai_format.append({"role": "assistant", "content": assistant})
    else:
        # 如果history为None，则设置为空列表
        history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="qwen-plus",
                                              messages=history_openai_format,
                                              temperature=1.0,)
                                              # stream=True)

    return response.choices[0].message.content

#定义API端点
@app.post("/",response_model=Response)
async def predict_api(prompt: Request):

    # 从实例化的Request对象中获取prompt属性
    prompt_value = prompt.prompt
    # 明确传递history参数，即使它是一个空列表
    response_text = await predict(prompt_value, history=[])
    return Response(response=response_text)

demo = gr.ChatInterface(
    fn = predict,
    title="表型 测试 Bot",
    theme="soft",
    description="请问我一些表型上的问题",
    examples=["大豆的生育期是多少？", "大豆的品种有哪些？", "大豆的种在哪里？"]
    )

#将Gradio界面嵌入到FastAPI应用程序中
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # mounting at the root path
    uvicorn.run(
        app="new_test2:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )