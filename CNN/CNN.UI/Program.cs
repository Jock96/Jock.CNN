namespace CNN.UI
{
    using System;
    using System.IO;

    using CNN.BL.Constants;
    using CNN.BL.Utils;

    using CNN.Core;

    /// <summary>
    /// Класс входной точки приложения.
    /// </summary>
    class Program
    {
        /// <summary>
        /// Точка входа приложения.
        /// </summary>
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.HELLO_MESSAGE);
            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);

            Console.ReadKey();

            Console.WriteLine(ConsoleMessageConstants.FILE_NAME_MESSAGE);

            var fileName = Console.ReadLine();
            var pathToResources = BL.Helpers.PathHelper.GetResourcesPath();

            var path = Path.Combine(pathToResources, fileName + FileConstants.IMAGE_EXTENSION);

            var converter = new ImageConverterUtil(path);
            var matrix = converter.ConvertImageToMatrix();

            var filterCore = FilterCore.Initialize();

            // TODO: Отладка, убрать.
            foreach (var value in filterCore)
                Console.WriteLine(value.ToString());

            Console.ReadKey();
        }
    }
}
