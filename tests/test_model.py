import os
import openai
import logging

logging.basicConfig(level=logging.INFO)

# OpenAI API配置
client = openai.OpenAI(
    api_key="sk-e572bd56dd184b8a994eda5b994f772a",
    base_url="https://api.deepseek.com/v1"
)

def test_model():
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "你是一个专业的数据质量评估专家，擅长分析和评估数据质量。"},
                {"role": "user", "content": "这是一个测试消息，请简单回复'测试成功'。"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        logging.info(f"API Response: {response}")
        logging.info(f"Content: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logging.error(f"Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    test_model()
