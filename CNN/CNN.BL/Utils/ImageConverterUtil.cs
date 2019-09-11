namespace CNN.BL.Utils
{
    using System.Drawing;
    using System.Threading;
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
            Bitmap image;

            try
            {
                image = new Bitmap(imagePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ConsoleMessageConstants.ERROR_MESSAGE + ex.ToString());
                Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);

                Console.ReadKey();

                Console.WriteLine(ConsoleMessageConstants.EXIT_MESSAGE);

                Thread.Sleep(500);
                Environment.Exit(0);

                return;
            }

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
        public double [,] ConvertImageToMatrix()
        {
            var size = _image.Size;
            var matrix = new double[size.Width, size.Height];
            
            // TODO: берём любой канал и на 255 делим.
            for (var xIndex = 0; xIndex < size.Width; ++xIndex)
                for (var yIndex = 0; yIndex < size.Height; ++yIndex)
                    matrix[xIndex, yIndex] = (_image.GetPixel(xIndex, yIndex).R / 255);

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

                var pathToSave = $"{newPath}{imageName}" +
                    $"{FileConstants.RESIZED_IMAGE_NAME_POSTFIX}{FileConstants.IMAGE_EXTENSION}";

                bitmap.Save(pathToSave);

                return bitmap;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE}{ex.ToString()}");
                return null;
            }
        }
    }
}
