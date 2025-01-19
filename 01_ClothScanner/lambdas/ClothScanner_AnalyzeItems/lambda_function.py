import json
import boto3
import base64
from io import BytesIO

def lambda_handler(event, context):

    system_prompt = """
    You will be given a photo of a person's outfit. 
    You will need to analyze and describe each fashion item in detail, and return as a JSON object that looks like <ExampleOutput>.
    Focus ONLY on fashion items, NOT the person and NOT the background of a photo.
    Think step by step.

    1. First find ALL fashion items.
    2. For each items, try to find all the <details> following. <details> Fabric, Color, Pattern, Style, Fit, Man or Women or Kids, Cut, Neckline, Sleeve Length, overall length, Closure, Details, Trim, Collar, Hem, Lining, Embellishments, Functionality, Brand Logo/Label. Gender and other things.</details> Fit is important.
    3. With the <details> above create a JSON object for each item with the following structure:
       {
         "color": "main color of the item",
         "category": one of ["TOP", "BOTTOM", "SHOES", "ACCESSORY", "OTHERS"],
         "details": "detailed description under 355 chars"
       }
    4. Provide the output as a JSON object with only one key: `items`. The `items` key should contain an array of the item JSON objects.

    <ExampleOutput>
    {
    "items": [
        {
        "color": "light blue",
        "category": "TOP",
        "details": "normal length, round neck neckline, long sleeve, normal silhouette, knit textile, plain print, casual style cardigan, with the word '86' written in the center"
        },
        {
        "color": "black",
        "category": "BOTTOM",
        "details": "high-waisted midi skirt with A-line silhouette, made from lightweight flowy fabric, features elastic waistband, falls just below the calf, suitable for casual to semi-formal wear"
        }
    ]
    }
    </ExampleOutput>
    """

    # S3 클라이언트 생성
    s3 = boto3.client('s3')
    # Bedrock 클라이언트 생성
    bedrock = boto3.client('bedrock-runtime')
    userid = event['UserID']
    date = event['Date']
    image_key = event['ImageKey']

    try:
        # S3에서 이미지 가져오기
        s3_response = s3.get_object(
            Bucket=event['BucketName'],
            Key=f"1_ClothScanner/{userid}/{date}/{image_key}"
            
        )
        
        # 이미지를 base64로 인코딩
        image_content = s3_response['Body'].read()
        base64_image = base64.b64encode(image_content).decode('utf-8')
        
        # Claude-3 Sonnet의 경우
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "system": system_prompt, 
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",  # 이미지 타입에 맞게 수정
                                "data": base64_image
                            }
                        },
                    ]
                }
            ]
        }

        # Bedrock 호출
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',  # 사용할 모델 ID
            body=json.dumps(payload)
        )
        
        # 응답 처리
        response_body = json.loads(response['body'].read())
        json_text = response_body['content'][0]['text']
        parsed_json = json.loads(json_text)  # 문자열을 JSON으로 파싱

        
        return {
            'statusCode': 200,
            'body': parsed_json
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }