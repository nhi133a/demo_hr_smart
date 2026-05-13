import boto3

session = boto3.Session()
credentials = session.get_credentials()

print("Access key:", credentials.access_key)
print("Token:", credentials.token)