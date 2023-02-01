package tasks

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotators.common.Annotated
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable

import scala.collection.Map

/** Tokenizes token into tokens by using camel case regex
 * 22Feb2023 ---> [22, Feb, 2023]
 * camelCaseTokenizer ---> [camel, Case, Tokenizer]
 *
 * Example
 *
 * import spark.implicits._
 * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
 * val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
 * val tokenizer = new Tokenizer().setInputCols("sentence").setOutputCol("token")
 * val camelCaseTokenizer = new CamelCaseTokenizer().setInputCols("token").setOutputCol("camelCase")
 * val pipeline = new Pipeline().setStages(Array(documentAssembler,sentenceDetector,tokenizer,camelCaseTokenizer))
 * val data = Seq("Date: 12Nov2022 " +
 *   "javaCodeExample: " +
 *   "public void savePatientRecord(String identityNo) { " +
 *   "PatientService patientService = new PatientService(); " +
 *   "Patient patient = patientService.getPatientByIdentityNo(identityNo)" +
 *   ".orElseThrow(() -> new RuntimeException(\"There is any patient with this identity number\")); " +
 *   "repository.persist(new PatientRecord(patient, LocalDateTime.now())); " +
 *   "} ").toDF("text")
 * val df = pipeline.fit(data).transform(data)
 * df.selectExpr("camelCase.result as camelCaseTokens").show(false)
 * df.selectExpr("token.result as tokens").show(false)
 * +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 * |camelCaseTokens                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
 * +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 * |[Date, :, 12, Nov, 2022, java, Code, Example, :, public, void, save, Patient, Record(String, identity, No, ), {, Patient, Service, patient, Service, =, new, Patient, Service, ();, Patient, patient, =, patient, Service.get, Patient, By, Identity, No(identity, No, )., or, Else, Throw, ((), -, >, new, Runtime, Exception("There, is, any, patient, with, this, identity, number, "));, repository.persist(new, Patient, Record(patient, ,, Local, Date, Time.now, ()));, }]|
 * +---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 * |tokens                                                                                                                                                                                                                                                                                                                                                                                                                               |
 * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 * |[Date, :, 12Nov2022, javaCodeExample, :, public, void, savePatientRecord(String, identityNo, ), {, PatientService, patientService, =, new, PatientService, ();, Patient, patient, =, patientService.getPatientByIdentityNo(identityNo, )., orElseThrow, ((), -, >, new, RuntimeException("There, is, any, patient, with, this, identity, number, "));, repository.persist(new, PatientRecord(patient, ,, LocalDateTime.now, ()));, }]|
 * +-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
 */

/** Params:
    uid – required uid for storing annotator to disk */
class CamelCaseTokenizer(override val uid: String)
    extends AnnotatorModel[CamelCaseTokenizer]
    with HasSimpleAnnotate[CamelCaseTokenizer] {

  /** Output annotator type: TOKEN */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type: TOKEN */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** overload constructor */
  def this() = this(Identifiable.randomUID("CAMEL_CASE_TOKENIZER"))

  /** turns number off , example: free3D -> free3D , Default: false */
  val disableNumbers: BooleanParam = new BooleanParam(this, "numbersState", "asdsa")

  /** turns number off , example: free3D -> free3D , Default: false */
  def setDisableNumbers(value: Boolean): this.type = set(disableNumbers, value)

  /** disableNumbers state */
  def getDisableNumbers(): Boolean = $(disableNumbers)

  val regexNumberOff = """(?<!(^|[A-Z]))(?=[A-Z])|(?<!^)(?=[A-Z][a-z])"""

  val regexAll = """(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[0-9])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=[0-9])"""

setDefault(disableNumbers -> false)

  /** one to many annotation
   * divides token into tokens according to camel case regex
   * returns: CamelCaseTokenizer -> Camel, Case, Tokenizer
   */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val indexedTokens = TokenSplit.unpack(annotations)
    val regex =  if ($(disableNumbers)) regexNumberOff.r  else regexAll.r
    val smallTokens = indexedTokens.map { token =>
      var currentPosition = 0
      val tokens = regex.split(token.content).map { x =>
        currentPosition = token.content.indexOf(x, currentPosition)
        val smallToken = SmallToken(x, currentPosition + token.start, currentPosition + token.start + x.length - 1, token.index)
        currentPosition += x.length
        smallToken
      }
        .filter(smallToken => smallToken.content.nonEmpty)
      tokens
    }
    TokenSplit.pack(smallTokens.flatten)
  }
}

/** Helper object to convert Annotations to SmallTokens, vice versa */
object TokenSplit extends Annotated[SmallToken]{
  override def annotatorType: String = TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[SmallToken] = {
    annotations
      .filter(_.annotatorType == annotatorType)
      .zipWithIndex.map { case (annotation, index) =>
      SmallToken(
        annotation.result,
        annotation.begin,
        annotation.end,
        index,
        Option(annotation.metadata))
    }
  }

  override def pack(items: Seq[SmallToken]): Seq[Annotation] = {
    items.map { item =>
      Annotation(
        annotatorType,
        item.start,
        item.end,
        item.content,
        Map("token" -> item.index.toString))
    }
  }
}

/** structure representing a part of token and its boundaries */
case class SmallToken(content: String,
                      start: Int,
                      end: Int,
                      index: Int,
                      metadata: Option[Map[String, String]] = None)