﻿namespace CNN.BL.Helpers
{
    using System.IO;

    /// <summary>
    /// Инструмент для работы с директориями.
    /// </summary>
    public static class PathHelper
    {
        /// <summary>
        /// Получает путь до папки ресурсов.
        /// </summary>
        /// <returns>Возвращает путь до папки ресурсов.</returns>
        public static string GetResourcesPath()
        {
            var rootPath = Directory.GetParent(Directory.GetParent(Directory.GetParent(Directory.GetCurrentDirectory()).FullName).FullName);
            return rootPath + Constants.FileConstants.BL_MODEL_NAME + $"{Constants.FileConstants.RESOURCES_PATH}";
        }
    }
}
