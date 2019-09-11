namespace CNN.BL.Constants
{
    using System.Collections.Generic;

    /// <summary>
    /// Класс констант диалогов.
    /// </summary>
    public static class DialogConstants
    {
        /// <summary>
        /// Положительные результаты.
        /// </summary>
        public static readonly List<string> YesResults = new List<string> { "Y", "y" };

        /// <summary>
        /// Отрицательные результаты.
        /// </summary>
        public static readonly List<string> NoResults = new List<string> { "N", "n" };

        /// <summary>
        /// Резуальтаты обучение.
        /// </summary>
        public static readonly List<string> LearnResults = new List<string> { "L", "l" };

        /// <summary>
        /// Резуальтаты распознавания.
        /// </summary>
        public static readonly List<string> RecognizeResults = new List<string> { "R", "R" };
    }
}
