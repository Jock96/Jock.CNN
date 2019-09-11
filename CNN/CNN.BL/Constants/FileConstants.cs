namespace CNN.BL.Constants
{
    /// <summary>
    /// Класс констант для работы с файлами.
    /// </summary>
    public static class FileConstants
    {
        /// <summary>
        /// Константа постфикса файла с изменённой размерностью.
        /// </summary>
        public const string RESIZED_IMAGE_NAME_POSTFIX = "(resized)";

        /// <summary>
        /// Директория сохранения весов.
        /// </summary>
        public const string WEIGHTS_DIRECTORY = "Weights";

        /// <summary>
        /// Директория ресурсов.
        /// </summary>
        public const string RESOURCES_PATH = "\\Resources";

        /// <summary>
        /// Имя модуля бизнес-логики.
        /// </summary>
        public const string BL_MODEL_NAME = "\\CNN.BL";

        /// <summary>
        /// Расширение файла изображения.
        /// </summary>
        public const string IMAGE_EXTENSION = ".bmp";

        /// <summary>
        /// Расширение текстового файла.
        /// </summary>
        public const string TEXT_EXTENSION = ".txt";
    }
}
