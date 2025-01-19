import json
import boto3
import base64
from datetime import datetime
import time

def save_to_s3(base64_image, bucket_name, key):
    try:
        s3 = boto3.client('s3')
        image_data = base64.b64decode(base64_image)
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=image_data,
            ContentType='image/png'
        )
        return key
    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        raise e

def lambda_handler(event, context):
    # Bedrock 클라이언트 생성
    bedrock = boto3.client('bedrock-runtime')
    
    try:
        # 입력 텍스트 가져오기
        user_id = event['user_id']
        date = event['date']
        bucket_name = event['bucket_name'] # S3 버킷 이름

        prompt = event['prompt']
        color = prompt['color']
        category = prompt['category']
        fashion_description = prompt['details']
        
        # 프롬프트 구성
        prompt = f"""3D icon of {color} colored {category}, {fashion_description}, Only front side. Flat color and lighting. Full shot. Center. Digital, grey background, design asset"""
    
        # Titan Image Generator 페이로드
        payload = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": "human model, busy background, multiple items"
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "quality": "standard",
                "height": 1024,
                "width": 1024
            }
        }

        # Bedrock 호출
        response = bedrock.invoke_model(
            modelId='amazon.titan-image-generator-v1',
            body=json.dumps(payload)
        )
        
        # 응답 처리
        response_body = json.loads(response['body'].read())
        
        # base64 이미지 추출 및 S3 저장
        generated_image = response_body.get('images', [])[0]
        timestamp = str(int(time.time()))
        image_key = f'1_ClothScanner/{user_id}/{date}/{category}_{timestamp}.png'
        s3_key = save_to_s3(generated_image, bucket_name, image_key)
        
        return {
            'statusCode': 200,
            'body': {
                'key': s3_key,
                'prompt': prompt
            }
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
