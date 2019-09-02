namespace CNN.BL.Utils
{
    using System.Drawing;

    using BL.Constants;
    using System;

    /// <summary>
    /// Класс конвертера изображений в матрицу.
    /// </summary>
    public class ImageConverterUtil
    {
        /// <summary>
        /// Инструмент конвертации изображений в матрицу.
        /// </summary>
        /// <param name="imagePath">Путь до изображения.</param>
        public ImageConverterUtil(string imagePath)
        {
            var image = new Bitmap(imagePath);

            if (image.Size.Height != MatrixConstants.MATRIX_SIZE ||
                image.Size.Width != MatrixConstants.MATRIX_SIZE)
                image = ResizeImage(imagePath, image);

            if (image == null)
                return;

            image.Dispose();
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
                using (var bitmap = new Bitmap(image, defaultSize))
                {
                    bitmap.Save($"{imagePath}{FileConstants.RESIZED_IMAGE_NAME_POSTFIX}.bmp");

                    return bitmap;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.ToString()}");
                return null;
            }
        }
    }
}
