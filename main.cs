using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XFeatures2D;
using System.IO;
using Emgu.CV.CvEnum;
using static System.Net.Mime.MediaTypeNames;

namespace opencv6
{
    public partial class main : Form
    {
        public main()
        {
            InitializeComponent();
            LoadReferenceImages();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            openFileDialog1.Filter = "Image files (*.jpg, *.jpeg, *.png) | *.jpg; *.jpeg; *.png";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                Mat inputImage = CvInvoke.Imread(openFileDialog1.FileName, ImreadModes.Color);

                // 设置pictureBox1的SizeMode属性以适应其宽度
                pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;

                Mat processedImage = PreProcess(inputImage);
                pictureBox1.Image = processedImage.ToBitmap();
                SaveProcessedImage(processedImage);

                string matchResult = PerformFeatureMatching(processedImage);
                textBox1.Text = matchResult;
            }
        }
        private Dictionary<string, Mat> _referenceImages = new Dictionary<string, Mat>();
        private SIFT _sift = new SIFT();

        private void LoadReferenceImages()
        {
            string referenceImagesPath = Path.Combine(Directory.GetCurrentDirectory(), "参考图片");
            var referenceImagesFiles = Directory.GetFiles(referenceImagesPath, "*.jpg");

            foreach (var file in referenceImagesFiles)
            {
                Mat image = CvInvoke.Imread(file, ImreadModes.Color);
                string imageName = Path.GetFileNameWithoutExtension(file);
                _referenceImages[imageName] = image;
            }
        }
        private Mat PreProcess(Mat inputImage)
        {
            // 1. 灰度转换
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(inputImage, grayImage, ColorConversion.Bgr2Gray);
            Mat equalizedImage = new Mat();
            CvInvoke.EqualizeHist(grayImage, equalizedImage);

            // 2. 缩放图片，指定高度为1000像素
            double scaleFactor = 1000.0 / inputImage.Height;
            Size newSize = new Size((int)(inputImage.Width * scaleFactor), 1000);
            Mat resizedImage = new Mat();
            CvInvoke.Resize(inputImage, resizedImage, newSize);

            // 3. 裁剪图片，起始位置为左上角 x+200, y+100，图片大小 800x800
            Rectangle roi1 = new Rectangle(200, 100, 800, 800);
            Mat croppedImage = new Mat(resizedImage, roi1);

            // 4. 对上步图片进行边缘检测，并对边缘进行绿色标识
            Mat edgeImage = new Mat();
            CvInvoke.Canny(croppedImage, edgeImage, 120, 200);
            Mat edgeOverlayImage = croppedImage.Clone();
            edgeOverlayImage.SetTo(new Bgr(0, 255, 0).MCvScalar, edgeImage);

            //// 5. 对上步图片进行霍夫圆变换，如果有相邻圆，仅保留一个，并对圆进行蓝色标识----不删除圆心偏移圆，速度快
            //CircleF[] circles = CvInvoke.HoughCircles(edgeImage, Emgu.CV.CvEnum.HoughModes.Gradient, 1, 100, 100, 50, 280, 400);
            //Mat circleOverlayImage = edgeOverlayImage.Clone();
            //foreach (CircleF circle in circles)
            //{
            //    CvInvoke.Circle(circleOverlayImage, new Point((int)circle.Center.X, (int)circle.Center.Y), (int)circle.Radius, new Bgr(255, 0, 0).MCvScalar, 2);
            //}


            // 5. 对上步图片进行霍夫圆变换，如果有相邻圆，仅保留一个，并对圆进行蓝色标识----删除圆心偏移圆，速度慢！！！！！！
            CircleF[] circles = CvInvoke.HoughCircles(edgeImage, Emgu.CV.CvEnum.HoughModes.Gradient, 1, 100, 100, 50, 280, 400);
            Mat circleOverlayImage = edgeOverlayImage.Clone();

            // 计算图像中心点
            int centerX = circleOverlayImage.Width / 2;
            int centerY = circleOverlayImage.Height / 2;

            // 记录保留的圆
            CircleF selectedCircle = new CircleF();
            bool hasSelectedCircle = false;

            foreach (CircleF circle in circles)
            {
                // 计算圆心到图像中心点的距离
                double distance = Math.Sqrt(Math.Pow(circle.Center.X - centerX, 2) + Math.Pow(circle.Center.Y - centerY, 2));

                if (distance <= 50)
                {
                    // 如果圆心在图像中心点半径50范围内，更新保留的圆
                    if (!hasSelectedCircle || circle.Radius > selectedCircle.Radius)
                    {
                        selectedCircle = circle;
                        hasSelectedCircle = true;
                    }
                }
            }

            if (hasSelectedCircle)
            {
                // 在输出图像中标记保留的圆
                CvInvoke.Circle(circleOverlayImage, new Point((int)selectedCircle.Center.X, (int)selectedCircle.Center.Y), (int)selectedCircle.Radius, new Bgr(255, 0, 0).MCvScalar, 2);
            }



            // 6. 根据此圆裁剪图片，同时将圆外部遮罩为白色
            Mat maskedImage = new Mat(circleOverlayImage.Size, DepthType.Cv8U, 3);
            maskedImage.SetTo(new MCvScalar(255, 255, 255));
            Mat mask = new Mat(circleOverlayImage.Size, DepthType.Cv8U, 1);
            if (circles.Length > 0)
            {
                mask.SetTo(new MCvScalar(0));
                CircleF circle = circles[0];
                CvInvoke.Circle(mask, new Point((int)circle.Center.X, (int)circle.Center.Y), (int)circle.Radius, new MCvScalar(255), -1);
                circleOverlayImage.CopyTo(maskedImage, mask);
            }
            else
            {
                circleOverlayImage.CopyTo(maskedImage);
            }
            return maskedImage;
        }
        private void SaveProcessedImage(Mat processedImage)
        {
            string matchResult2 = PerformFeatureMatching(processedImage);
            string savePath = Path.Combine(Directory.GetCurrentDirectory(), "处理结果", $"{matchResult2 + "-" + DateTime.Now.ToString("yyyyMMddHHmmss")}.jpg");
            CvInvoke.Imwrite(savePath, processedImage);
        }

        private string PerformFeatureMatching(Mat inputImage)
        {
            string bestMatch = string.Empty;
            float bestMatchScore = float.MaxValue;

            foreach (var referenceImage in _referenceImages)
            {
                float matchScore = MatchImagesUsingORB(inputImage, referenceImage.Value);

                if (matchScore < bestMatchScore)
                {
                    bestMatchScore = matchScore;
                    bestMatch = referenceImage.Key;
                }
            }

            return bestMatch;
        }

        private float MatchImagesUsingORB(Mat img1, Mat img2)
        {
            // 创建一个ORB特征检测器实例
            ORBDetector orb = new ORBDetector();

            VectorOfKeyPoint keyPoints1 = new VectorOfKeyPoint();
            VectorOfKeyPoint keyPoints2 = new VectorOfKeyPoint();
            Mat descriptors1 = new Mat();
            Mat descriptors2 = new Mat();

            // 使用ORB检测关键点和计算描述符
            orb.DetectAndCompute(img1, null, keyPoints1, descriptors1, false);
            orb.DetectAndCompute(img2, null, keyPoints2, descriptors2, false);

            // 更改距离类型为Hamming
            BFMatcher matcher = new BFMatcher(DistanceType.Hamming);
            VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch();
            matcher.Add(descriptors1);
            matcher.KnnMatch(descriptors2, matches, 2);

            float sumDistances = 0;
            int numGoodMatches = 0;

            foreach (var match in matches.ToArrayOfArray())
            {
                if (match[0].Distance < 0.85 * match[1].Distance)
                {
                    sumDistances += match[0].Distance;
                    numGoodMatches++;
                }
            }
            return sumDistances / numGoodMatches;
        }

        private void main_Load(object sender, EventArgs e)
        {

        }

    }
}
