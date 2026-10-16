package ai.liquid.lfmdemo.glasses

/**
 * Text normalization for the Chinese sherpa TTS engine: its lexicon has no
 * readings for ASCII digits or letters, so "1加1等于2" is spoken as "加等于"
 * and "ABCD后面字母是E" as "后面字母是". Convert ASCII numbers to Han
 * readings and contiguous ASCII letter runs to per-letter readings.
 * Numbers <= 4 digits are read as values (三百 / 一千零一); >= 5 digits and
 * 2xxx years fall back to digit-by-digit (phone, serial, year style).
 */
internal object ChineseTtsNormalizer {

    private val DIGITS = arrayOf("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")
    private val UNITS = arrayOf("", "十", "百", "千")
    private val BIG_UNITS = arrayOf("", "万", "亿", "万亿")
    private val NUMBER = Regex("[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?%?")
    private val YEAR = Regex("([12]\\d{3})年")
    private val LETTER_RUN = Regex("[A-Za-z]+")

    /** Common whole-word readings preferred over spelling letter by letter. */
    private val LETTER_WORDS = mapOf(
        "ok" to "欧开", "ai" to "诶爱", "wifi" to "歪法", "app" to "阿普",
        "usb" to "优艾斯比", "cpu" to "西批优", "gpu" to "鸡批优",
        "id" to "爱迪", "vr" to "威阿", "ar" to "诶阿", "gps" to "鸡批艾斯",
        "qq" to "扣扣", "win" to "温", "tts" to "踢踢艾斯", "asr" to "诶艾斯阿",
        "llm" to "艾勒艾勒艾姆", "mb" to "艾姆比", "gb" to "吉比", "kb" to "开比"
    )

    private val LETTER_READINGS = mapOf(
        'a' to "诶", 'b' to "比", 'c' to "西", 'd' to "迪", 'e' to "伊",
        'f' to "艾夫", 'g' to "吉", 'h' to "艾曲", 'i' to "爱", 'j' to "杰",
        'k' to "开", 'l' to "艾勒", 'm' to "艾姆", 'n' to "恩", 'o' to "欧",
        'p' to "批", 'q' to "扣", 'r' to "阿", 's' to "艾斯", 't' to "踢",
        'u' to "优", 'v' to "威", 'w' to "达布优", 'x' to "艾克斯", 'y' to "歪",
        'z' to "贼"
    )

    fun normalize(text: String): String {
        val yearsReplaced = YEAR.replace(text) { match ->
            match.groupValues[1].map { DIGITS[it - '0'] }.joinToString("") + "年"
        }
        // A whole number directly attached to a letter run (RTX4090,
        // abc123, 64K) is a code/serial, not a quantity: read it digit by
        // digit. Decimals (0.5Km) stay numeric readings.
        val codeNumbers = Regex("(?<=[A-Za-z])\\d+(?![.\\d]|\\d)|(?:^|(?![.\\d]))\\d+(?=[A-Za-z])").replace(yearsReplaced) { match ->
            match.value.map { DIGITS[it - '0'] }.joinToString("")
        }
        val numbersReplaced = NUMBER.replace(codeNumbers) { match -> convert(match.value) }
        return LETTER_RUN.replace(numbersReplaced) { match ->
            val run = match.value
            LETTER_WORDS[run.lowercase()] ?: run.map { letter ->
                LETTER_READINGS[letter.lowercaseChar()] ?: letter.toString()
            }.joinToString("")
        }
    }

    private fun convert(raw: String): String {
        var number = raw
        var suffix = ""
        if (number.endsWith("%")) {
            suffix = "百分之"
            number = number.dropLast(1)
        }
        val negative = number.startsWith("-")
        if (number.startsWith("+") || number.startsWith("-")) number = number.substring(1)
        val integerPart = number.split(".")[0].replace(",", "")
        val fractionPart = if (number.contains(".")) number.split(".", limit = 2)[1] else null
        val sign = if (negative) "负" else ""
        val integer = if (integerPart.length >= 5 || integerPart.all { it == '0' }) {
            digitsToHan(integerPart)
        } else {
            integerValueToHan(integerPart.toIntOrNull() ?: return raw)
        }
        val fraction = fractionPart?.let { "点" + it.map { DIGITS[it - '0'] }.joinToString("") } ?: ""
        return sign + suffix + integer + fraction
    }

    /** Digit-by-digit reading for long strings (dates, phone numbers). */
    private fun digitsToHan(digits: String): String =
        digits.map { DIGITS[it - '0'] }.joinToString("")

    /** Value reading for small integers (1..9999): 二百三十四, 十五, 一千零一. */
    private fun integerValueToHan(value: Int): String {
        if (value == 0) return "零"
        var remaining = value
        var magnitude = 0
        val groups = ArrayList<String>()
        while (remaining > 0) {
            val group = remaining % 10000
            if (group != 0) {
                groups.add(groupToHan(group) + BIG_UNITS[magnitude])
            } else if (groups.isNotEmpty() && groups.last().first() != '零') {
                groups.add("零")
            }
            remaining /= 10000
            magnitude += 1
        }
        return groups.asReversed().joinToString("")
    }

    private fun groupToHan(group: Int): String {
        val result = StringBuilder()
        var rest = group
        var unitIndex = 3
        var pendingZero = false
        while (unitIndex >= 0) {
            val digit = rest / pow10(unitIndex)
            rest %= pow10(unitIndex)
            if (digit == 0) {
                if (result.isNotEmpty()) pendingZero = true
            } else {
                if (pendingZero) {
                    result.append("零")
                    pendingZero = false
                }
                // 10..19 reads 十/十五, not 一十/一十五
                val skipOne = digit == 1 && unitIndex == 1 && result.isEmpty() && group < 100
                if (!skipOne) result.append(DIGITS[digit])
                result.append(UNITS[unitIndex])
            }
            unitIndex -= 1
        }
        return result.toString()
    }

    private fun pow10(exponent: Int): Int = when (exponent) {
        0 -> 1; 1 -> 10; 2 -> 100; else -> 1000
    }
}
