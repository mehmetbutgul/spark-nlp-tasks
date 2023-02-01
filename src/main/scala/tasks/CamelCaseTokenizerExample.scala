package tasks

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import org.apache.spark.ml.Pipeline

object CamelCaseTokenizerExample extends App {
  val spark = SparkUtil.getSession()
  import spark.implicits._

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text").setOutputCol("document")

  val sentenceDetector = new SentenceDetector()
    .setInputCols("document").setOutputCol("sentence")

  val tokenizer = new Tokenizer()
    .setInputCols("sentence").setOutputCol("token")

  val camelCaseTokenizer = new CamelCaseTokenizer()
    .setInputCols("token").setOutputCol("camelCase")

  val pipeline = new Pipeline()
    .setStages(Array(documentAssembler,sentenceDetector,tokenizer,camelCaseTokenizer))

  val data = Seq("Date: 12Nov2022 " +
    "javaCodeExample: " +
    "public void savePatientRecord(String identityNo) { " +
    /*"PatientService patientService = new PatientService(); " +
    "Patient patient = patientService.getPatientByIdentityNo(identityNo)" +
    ".orElseThrow(() -> new RuntimeException(\"There is any patient with this identity number\")); " +
    "repository.persist(new PatientRecord(patient, LocalDateTime.now())); " +*/
    "} ").toDF("text")

  val df = pipeline.fit(data).transform(data)
  df.selectExpr("token").show(false)
  df.selectExpr("camelCase").show(false)
}
