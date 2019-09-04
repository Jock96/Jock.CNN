namespace CNN.BL.Enums
{
    /// <summary>
    /// Перечисление типов слоёв нейронной сети.
    /// </summary>
    public enum LayerType
    {
        /// <summary>
        /// Входной слой.
        /// </summary>
        Input,

        /// <summary>
        /// Свёрточный слой.
        /// </summary>
        Convolutional,

        /// <summary>
        /// Скрытый слой.
        /// </summary>
        Hidden,

        /// <summary>
        /// Выходной слой.
        /// </summary>
        Output,

        /// <summary>
        /// Неизвестный тип слоя.
        /// </summary>
        None
    }
}
