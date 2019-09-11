namespace CNN.BL.Utils
{
    using System.Drawing;
    using System.Threading;
    using System.IO;
    using System;

    using BL.Constants;
    using System.Collections.Generic;

    /// <summary>
    /// Класс конвертера изображений в матрицу.
    /// </summary>
    public class ImageConverterUtil
    {
        /// <summary>
        /// Полученное изображение.
        /// </summary>
        private List<Bitmap> _images;

        /// <summary>
        /// Инструмент конвертации изображений в матрицу.
        /// </summary>
        /// <param name="imagePathes">Пути до изображения.</param>
        public ImageConverterUtil(List<string> imagePathes)
        {
            _images = new List<Bitmap>();
            var pathToImagesDictionary = new Dictionary<string, Bitmap>();

            foreach(var imagePath in imagePathes)
            {
                try
                {
                    pathToImagesDictionary.Add(imagePath, new Bitmap(imagePath));
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
            }

            foreach (var keyValuePair in pathToImagesDictionary)
            {
                if (keyValuePair.Value.Size.Height != MatrixConstants.MATRIX_SIZE ||
                keyValuePair.Value.Size.Width != MatrixConstants.MATRIX_SIZE)
                {
                    var convertedImage = ResizeImage(keyValuePair.Key, keyValuePair.Value);

                    if (convertedImage == null)
                        return;

                    _images.Add(convertedImage);
                }
                else
                {
                    _images.Add(keyValuePair.Value);
                }
            }
        }

        /// <summary>
        /// Конвертировать изображение в матрицу значений.
        /// </summary>
        /// <returns>Возвращает матрицу значений.</returns>
        public List<double[,]> ConvertImagesToMatrix()
        {
            var listOfMatrix = new List<double[,]>();

            foreach (var image in _images)
            {
                var size = image.Size;
                var matrix = new double[size.Width, size.Height];

                // TODO: берём любой канал и на 255 делим.
                for (var xIndex = 0; xIndex < size.Width; ++xIndex)
                    for (var yIndex = 0; yIndex < size.Height; ++yIndex)
                        matrix[xIndex, yIndex] = (image.GetPixel(xIndex, yIndex).R / 255);

                listOfMatrix.Add(matrix);
            }

            return listOfMatrix;
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

            var imageName = Path.GetFileNameWithoutExtension(imagePath);

            var imageNameWithExtension = Path.GetFileName(imagePath);
            var pathWithoutFile = imagePath.Replace(imageNameWithExtension, string.Empty);

            var directoryToSave = Path.Combine(pathWithoutFile, FileConstants.RESIZED_IMAGE_NAME_POSTFIX);

            if (!Directory.Exists(directoryToSave))
                Directory.CreateDirectory(directoryToSave);

            try
            {
                var bitmap = new Bitmap(image, defaultSize);

                var pathToSave = $"{directoryToSave}\\{imageName}" +
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
