
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.size
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import tasks.CamelCaseTokenizer

class CamelCaseTokenizerTestSpec extends AnyFlatSpec {
  val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val sentence: SentenceDetector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  "CamelCaseTokenizer" should "correctly tokenize camel case tokens from tokenizer's results" taggedAs Tag("Test") in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(1 -> ("Date: 12Nov2022 " +
          "javaCodeExample: " +
          "public void savePatientRecord(String identityNo) { " +
          "} "))).toDF("id", "text")

    val expectedCamelCaseTokens = Seq(
      Annotation(TOKEN, 0, 3, "Date", Map("token" -> "0")),
      Annotation(TOKEN, 4, 4, ":", Map("token" -> "1")),
      Annotation(TOKEN, 6, 7, "12", Map("token" -> "2")),
      Annotation(TOKEN, 8, 10, "Nov", Map("token" -> "2")),
      Annotation(TOKEN, 11, 14, "2022", Map("token" -> "2")),
      Annotation(TOKEN, 16, 19, "java", Map("token" -> "3")),
      Annotation(TOKEN, 20, 23, "Code", Map("token" -> "3")),
      Annotation(TOKEN, 24, 30, "Example", Map("token" -> "3")),
      Annotation(TOKEN, 31, 31, ":", Map("token" -> "4")),
      Annotation(TOKEN, 33, 38, "public", Map("token" -> "5")),
      Annotation(TOKEN, 40, 43, "void", Map("token" -> "6")),
      Annotation(TOKEN, 45, 48, "save", Map("token" -> "7")),
      Annotation(TOKEN, 49, 55, "Patient", Map("token" -> "7")),
      Annotation(TOKEN, 56, 68, "Record(String", Map("token" -> "7")),
      Annotation(TOKEN, 70, 77, "identity", Map("token" -> "8")),
      Annotation(TOKEN, 78, 79, "No", Map("token" -> "8")),
      Annotation(TOKEN, 80, 80, ")", Map("token" -> "9")),
      Annotation(TOKEN, 82, 82, "{", Map("token" -> "10")),
      Annotation(TOKEN, 84, 84, "}", Map("token" -> "11")))

    val camelCaseTokenizer = new CamelCaseTokenizer()
      .setInputCols("token")
      .setOutputCol("camelCaseTokens")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, camelCaseTokenizer))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select(size(pipelineDF("token.result")).as("totalTokens")).show
    pipelineDF.select(size(pipelineDF("camelCaseTokens.result")).as("totalCamelCaseTokens")).show

    val camelCaseTokens = Annotation.collect(pipelineDF, "camelCaseTokens").flatten.toSeq

    assert(camelCaseTokens == expectedCamelCaseTokens)
  }
}