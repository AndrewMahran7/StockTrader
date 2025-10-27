#!/bin/bash

# AWS EC2 Deployment Script for ORB Trading Bot
# This script sets up a complete trading environment on EC2

set -e

echo "🚀 Starting AWS EC2 deployment for ORB Trading Bot..."

# Configuration
INSTANCE_TYPE="t3.small"
REGION="us-east-1"
KEY_NAME="trading-bot-key"
SECURITY_GROUP="trading-bot-sg"
IMAGE_ID="ami-0c55b159cbfafe1d0"  # Amazon Linux 2

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first:"
    echo "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi

print_success "AWS CLI configured and credentials verified"

# Create key pair if it doesn't exist
print_status "Creating SSH key pair..."
if ! aws ec2 describe-key-pairs --key-names $KEY_NAME --region $REGION &> /dev/null; then
    aws ec2 create-key-pair --key-name $KEY_NAME --region $REGION --query 'KeyMaterial' --output text > ${KEY_NAME}.pem
    chmod 400 ${KEY_NAME}.pem
    print_success "SSH key pair created: ${KEY_NAME}.pem"
else
    print_success "SSH key pair already exists"
fi

# Create security group
print_status "Creating security group..."
if ! aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION &> /dev/null; then
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
        --group-name $SECURITY_GROUP \
        --description "Security group for ORB Trading Bot" \
        --region $REGION \
        --query 'GroupId' \
        --output text)
    
    # Allow SSH access
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    # Allow HTTP access (for dashboard)
    aws ec2 authorize-security-group-ingress \
        --group-id $SECURITY_GROUP_ID \
        --protocol tcp \
        --port 5000 \
        --cidr 0.0.0.0/0 \
        --region $REGION
    
    print_success "Security group created: $SECURITY_GROUP_ID"
else
    SECURITY_GROUP_ID=$(aws ec2 describe-security-groups --group-names $SECURITY_GROUP --region $REGION --query 'SecurityGroups[0].GroupId' --output text)
    print_success "Security group already exists: $SECURITY_GROUP_ID"
fi

# Launch EC2 instance
print_status "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $IMAGE_ID \
    --count 1 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP_ID \
    --region $REGION \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ORB-Trading-Bot}]' \
    --user-data file://ec2-user-data.sh \
    --query 'Instances[0].InstanceId' \
    --output text)

print_success "EC2 instance launched: $INSTANCE_ID"

# Wait for instance to be running
print_status "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

print_success "Instance is running at IP: $PUBLIC_IP"

# Wait for instance to be ready
print_status "Waiting for instance to be ready for SSH..."
sleep 60

echo ""
echo "🎉 Deployment Complete!"
echo ""
echo "📋 Instance Details:"
echo "   Instance ID: $INSTANCE_ID"
echo "   Public IP: $PUBLIC_IP"
echo "   SSH Key: ${KEY_NAME}.pem"
echo ""
echo "🔧 Next Steps:"
echo "1. SSH into the instance:"
echo "   ssh -i ${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
echo ""
echo "2. Upload your .env file with Alpaca API credentials:"
echo "   scp -i ${KEY_NAME}.pem .env ec2-user@$PUBLIC_IP:/home/ec2-user/trading/"
echo ""
echo "3. Access the trading dashboard:"
echo "   http://$PUBLIC_IP:5000"
echo ""
echo "⚠️  Security Notes:"
echo "   - The dashboard is publicly accessible. Consider adding IP restrictions."
echo "   - Store your SSH key (${KEY_NAME}.pem) securely."
echo "   - Monitor your AWS costs and trading activity."
echo ""
echo "📊 Monitoring:"
echo "   - View logs: ssh to instance and run 'docker logs trading-bot'"
echo "   - Check status: ssh to instance and run 'docker ps'"
echo ""