namespace CNN.BL.Utils
{
    using System.Drawing;
    using System.IO;
    using System;

    using BL.Constants;

    /// <summary>
    /// Класс конвертера изображений в матрицу.
    /// </summary>
    public class ImageConverterUtil
    {
        /// <summary>
        /// Полученное изображение.
        /// </summary>
        private Bitmap _image;

        /// <summary>
        /// Инструмент конвертации изображений в матрицу.
        /// </summary>
        /// <param name="imagePath">Путь до изображения.</param>
        public ImageConverterUtil(string imagePath)
        {
            var image = new Bitmap(imagePath);

            if (image.Size.Height != MatrixConstants.MATRIX_SIZE ||
                image.Size.Width != MatrixConstants.MATRIX_SIZE)
            {
                var convertedImage = ResizeImage(imagePath, image);

                if (convertedImage == null)
                    return;

                _image = convertedImage;
            }
            else
            {
                _image = image;
            }
        }

        /// <summary>
        /// Конвертировать изображение в матрицу значений.
        /// </summary>
        /// <returns>Возвращает матрицу значений.</returns>
        public int [,] ConvertImageToMatrix()
        {
            var size = _image.Size;
            var matrix = new int[size.Width, size.Height];

            for (var xIndex = 0; xIndex < size.Width; ++xIndex)
                for (var yIndex = 0; yIndex < size.Height; ++yIndex)
                    matrix[xIndex, yIndex] = _image.GetPixel(xIndex, yIndex).ToArgb();

            return matrix;
        }

        /// <summary>
        /// Изменяет размерность изображения так, 
        /// чтобы соответствовать размеру по-умолчанию.
        /// </summary>
        /// <param name="imagePath">Путь до изображения.</param>
        /// <param name="image">Изображение.</param>
        /// <returns>Возвращает изображение с изменённой размерностью.</returns>
        private static Bitmap ResizeImage(string imagePath, Bitmap image)
        {
            var defaultSize = new Size(MatrixConstants.MATRIX_SIZE, MatrixConstants.MATRIX_SIZE);

            try
            {
                var bitmap = new Bitmap(image, defaultSize);
                var imageName = Path.GetFileNameWithoutExtension(imagePath);

                var imageNameWithExtension = Path.GetFileName(imagePath);
                var newPath = imagePath.Replace(imageNameWithExtension, string.Empty);

                image.Save($"{newPath}{imageName}{FileConstants.RESIZED_IMAGE_NAME_POSTFIX}",
                    System.Drawing.Imaging.ImageFormat.Bmp);

                return bitmap;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.ToString()}");
                return null;
            }
        }
    }
}
