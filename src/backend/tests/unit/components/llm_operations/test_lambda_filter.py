import contextlib
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lfx.components.llm_operations.lambda_filter import LambdaFilterComponent
from lfx.schema import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.utils.sandbox import SandboxResult, SandboxUnavailableError

from tests.base import ComponentTestBaseWithoutClient


class TestLambdaFilterComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return LambdaFilterComponent

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that works with async invoke."""
        mock = AsyncMock()
        mock.ainvoke = AsyncMock()
        return mock

    @pytest.fixture
    def model_metadata(self):
        """Helper fixture that returns standard model metadata structure."""
        return [
            {
                "name": "gpt-3.5-turbo",
                "provider": "OpenAI",
                "metadata": {
                    "model_class": "MockLanguageModel",
                    "model_name_param": "model",
                    "api_key_param": "api_key",
                },
            }
        ]

    @pytest.fixture
    def default_kwargs(self, model_metadata):
        """Return the default kwargs for the component with proper model metadata."""
        return {
            "data": [Data(data={"items": [{"name": "test1", "value": 10}, {"name": "test2", "value": 20}]})],
            "model": model_metadata,
            "api_key": "test-api-key",
            "filter_instruction": "Filter items with value greater than 15",
            "sample_size": 1000,
            "max_size": 30000,
        }

    @pytest.fixture
    def file_names_mapping(self):
        return []

    @pytest.fixture(autouse=True)
    def _sandbox_off_by_default(self):
        """Pin the sandbox off so these tests assert the in-process path deterministically.

        Without this a developer or CI runner that has LANGFLOW_SANDBOX_BACKEND
        configured would route them into a real backend. Sandbox tests below opt
        back in explicitly.
        """
        with patch("lfx.components.llm_operations.lambda_filter.is_sandbox_enabled", return_value=False):
            yield


class TestValidateLambda(TestLambdaFilterComponent):
    """Tests for _validate_lambda method."""

    def test_should_return_true_when_lambda_is_valid(self, component_class):
        # Arrange
        component = component_class()
        valid_lambda = "lambda x: x + 1"

        # Act
        result = component._validate_lambda(valid_lambda)

        # Assert
        assert result is True

    def test_should_return_false_when_lambda_keyword_missing(self, component_class):
        # Arrange
        component = component_class()
        invalid_lambda = "x: x + 1"

        # Act
        result = component._validate_lambda(invalid_lambda)

        # Assert
        assert result is False

    def test_should_return_false_when_colon_missing(self, component_class):
        # Arrange
        component = component_class()
        invalid_lambda = "lambda x x + 1"

        # Act
        result = component._validate_lambda(invalid_lambda)

        # Assert
        assert result is False

    def test_should_return_true_when_lambda_has_whitespace(self, component_class):
        # Arrange
        component = component_class()
        valid_lambda = "  lambda x: x + 1  "

        # Act
        result = component._validate_lambda(valid_lambda)

        # Assert
        assert result is True


class TestParseLambdaSandbox(TestLambdaFilterComponent):
    """The LLM-generated lambda is untrusted (prompt-injection) and must be sandboxed."""

    def test_benign_lambda_still_works(self, component_class):
        component = component_class()
        fn = component._parse_lambda_from_response("lambda x: x + 1")
        assert fn(41) == 42

    def test_benign_lambda_can_use_safe_builtins(self, component_class):
        component = component_class()
        fn = component._parse_lambda_from_response("lambda x: len(x)")
        assert fn([1, 2, 3]) == 3

    def test_dunder_escape_gadget_is_rejected_at_parse(self, component_class):
        """Dunder attribute traversal is rejected up front by the AST safety check."""
        component = component_class()
        with pytest.raises(ValueError, match="unsafe lambda"):
            component._parse_lambda_from_response("lambda x: x.__class__.__bases__")

    def test_import_builtin_is_unreachable_at_call(self, component_class):
        """`__import__` is a builtin Name, so the curated builtins make it raise NameError when called."""
        component = component_class()
        fn = component._parse_lambda_from_response("lambda x: __import__('os').system('id')")
        with pytest.raises(NameError):
            fn("ignored")

    def test_open_builtin_is_unreachable(self, component_class):
        """`open` is absent from the curated builtins, so the lambda raises NameError at call time."""
        component = component_class()
        fn = component._parse_lambda_from_response("lambda x: open('/etc/passwd')")
        with pytest.raises(NameError):
            fn("ignored")


class TestGetDataStructure(TestLambdaFilterComponent):
    """Tests for get_data_structure method."""

    def test_should_return_type_name_when_input_is_primitive(self, component_class):
        # Arrange
        component = component_class()
        bool_value = True

        # Act & Assert
        assert component.get_data_structure("test") == "str"
        assert component.get_data_structure(42) == "int"
        assert component.get_data_structure(3.14) == "float"
        assert component.get_data_structure(bool_value) == "bool"

    def test_should_return_dict_structure_when_input_is_dict(self, component_class):
        # Arrange
        component = component_class()
        test_data = {"key": "value", "number": 42}

        # Act
        result = component.get_data_structure(test_data)

        # Assert
        assert result == {"key": "str", "number": "int"}

    def test_should_return_list_structure_when_input_is_list(self, component_class):
        # Arrange
        component = component_class()
        test_data = [1, 2, 3]

        # Act
        result = component.get_data_structure(test_data)

        # Assert
        assert result == ["int"]

    def test_should_return_empty_list_when_input_is_empty_list(self, component_class):
        # Arrange
        component = component_class()

        # Act
        result = component.get_data_structure([])

        # Assert
        assert result == []

    def test_should_return_nested_structure_when_input_is_nested(self, component_class):
        # Arrange
        component = component_class()
        test_data = {"nested": {"a": [{"b": 1}]}}

        # Act
        result = component.get_data_structure(test_data)

        # Assert
        assert result == {"nested": {"a": [{"b": "int"}]}}


class TestGetInputTypeName(TestLambdaFilterComponent):
    """Tests for _get_input_type_name method."""

    def test_should_return_message_when_input_is_single_message(self, component_class):
        # Arrange
        component = component_class()
        component.data = Message(text="test")

        # Act
        result = component._get_input_type_name()

        # Assert
        assert result == "Message"

    def test_should_return_message_when_input_is_list_of_messages(self, component_class):
        # Arrange
        component = component_class()
        component.data = [Message(text="test1"), Message(text="test2")]

        # Act
        result = component._get_input_type_name()

        # Assert
        assert result == "Message"

    def test_should_return_dataframe_when_input_is_dataframe(self, component_class):
        # Arrange
        component = component_class()
        component.data = DataFrame([{"a": 1}])

        # Act
        result = component._get_input_type_name()

        # Assert
        assert result == "DataFrame"

    def test_should_return_data_when_input_is_data(self, component_class):
        # Arrange
        component = component_class()
        component.data = Data(data={"key": "value"})

        # Act
        result = component._get_input_type_name()

        # Assert
        assert result == "Data"

    def test_should_return_unknown_when_input_is_empty_list(self, component_class):
        # Arrange
        component = component_class()
        component.data = []

        # Act
        result = component._get_input_type_name()

        # Assert
        assert result == "unknown"


class TestIsMessageInput(TestLambdaFilterComponent):
    """Tests for _is_message_input method."""

    def test_should_return_true_when_input_is_single_message(self, component_class):
        # Arrange
        component = component_class()
        component.data = Message(text="test")

        # Act
        result = component._is_message_input()

        # Assert
        assert result is True

    def test_should_return_true_when_input_is_list_of_messages(self, component_class):
        # Arrange
        component = component_class()
        component.data = [Message(text="test1"), Message(text="test2")]

        # Act
        result = component._is_message_input()

        # Assert
        assert result is True

    def test_should_return_false_when_input_is_data(self, component_class):
        # Arrange
        component = component_class()
        component.data = Data(data={"key": "value"})

        # Act
        result = component._is_message_input()

        # Assert
        assert result is False

    def test_should_return_false_when_input_is_empty_list(self, component_class):
        # Arrange
        component = component_class()
        component.data = []

        # Act
        result = component._is_message_input()

        # Assert
        assert result is False


class TestExtractMessageText(TestLambdaFilterComponent):
    """Tests for _extract_message_text method."""

    def test_should_return_text_when_input_is_single_message(self, component_class):
        # Arrange
        component = component_class()
        component.data = Message(text="Hello World")

        # Act
        result = component._extract_message_text()

        # Assert
        assert result == "Hello World"

    def test_should_return_empty_string_when_message_text_is_none(self, component_class):
        # Arrange
        component = component_class()
        component.data = Message(text=None)

        # Act
        result = component._extract_message_text()

        # Assert
        assert result == ""

    def test_should_join_texts_when_input_is_list_of_messages(self, component_class):
        # Arrange
        component = component_class()
        component.data = [Message(text="Hello"), Message(text="World")]

        # Act
        result = component._extract_message_text()

        # Assert
        assert result == "Hello\n\nWorld"

    def test_should_return_single_text_when_list_has_one_message(self, component_class):
        # Arrange
        component = component_class()
        component.data = [Message(text="Only one")]

        # Act
        result = component._extract_message_text()

        # Assert
        assert result == "Only one"


class TestExtractStructuredData(TestLambdaFilterComponent):
    """Tests for _extract_structured_data method."""

    def test_should_return_dict_when_input_is_single_data(self, component_class):
        # Arrange
        component = component_class()
        component.data = Data(data={"key": "value"})

        # Act
        result = component._extract_structured_data()

        # Assert
        assert result == {"key": "value"}

    def test_should_return_records_when_input_is_dataframe(self, component_class):
        # Arrange
        component = component_class()
        component.data = DataFrame([{"a": 1}, {"a": 2}])

        # Act
        result = component._extract_structured_data()

        # Assert
        assert result == [{"a": 1}, {"a": 2}]

    def test_should_combine_data_when_input_is_list_of_data(self, component_class):
        # Arrange
        component = component_class()
        component.data = [Data(data={"a": 1}), Data(data={"b": 2})]

        # Act
        result = component._extract_structured_data()

        # Assert
        assert result == [{"a": 1}, {"b": 2}]

    def test_should_unwrap_single_dict_when_list_has_one_item(self, component_class):
        # Arrange
        component = component_class()
        component.data = [Data(data={"only": "one"})]

        # Act
        result = component._extract_structured_data()

        # Assert
        assert result == {"only": "one"}

    def test_should_return_empty_dict_when_no_data_extracted(self, component_class):
        # Arrange
        component = component_class()
        component.data = []

        # Act
        result = component._extract_structured_data()

        # Assert
        assert result == {}


class TestBuildTextPrompt(TestLambdaFilterComponent):
    """Tests for _build_text_prompt method."""

    def test_should_include_full_text_when_text_is_small(self, component_class):
        # Arrange
        component = component_class()
        component.max_size = 1000
        component.sample_size = 100
        component.filter_instruction = "Transform to uppercase"
        text = "Short text"

        # Act
        result = component._build_text_prompt(text)

        # Assert
        assert "Short text" in result
        assert "Transform to uppercase" in result

    def test_should_truncate_text_when_text_is_large(self, component_class):
        # Arrange
        component = component_class()
        component.max_size = 50
        component.sample_size = 10
        component.filter_instruction = "Summarize"
        text = "A" * 100

        # Act
        result = component._build_text_prompt(text)

        # Assert
        assert "Text length: 100 characters" in result
        assert "First 10 characters" in result
        assert "Last 10 characters" in result


class TestBuildDataPrompt(TestLambdaFilterComponent):
    """Tests for _build_data_prompt method."""

    def test_should_include_full_data_when_data_is_small(self, component_class):
        # Arrange
        component = component_class()
        component.max_size = 1000
        component.sample_size = 100
        component.filter_instruction = "Filter by value"
        data = {"key": "value"}

        # Act
        result = component._build_data_prompt(data)

        # Assert
        assert '"key": "value"' in result
        assert "Filter by value" in result

    def test_should_truncate_data_when_data_is_large(self, component_class):
        # Arrange
        component = component_class()
        component.max_size = 50
        component.sample_size = 10
        component.filter_instruction = "Filter"
        data = {"key": "A" * 100}

        # Act
        result = component._build_data_prompt(data)

        # Assert
        assert "Data is too long to display" in result
        assert "First lines (head)" in result
        assert "Last lines (tail)" in result


class TestConvertResultToData(TestLambdaFilterComponent):
    """Tests for _convert_result_to_data method."""

    def test_should_wrap_dict_when_result_is_dict(self, component_class):
        # Arrange
        component = component_class()
        result = {"key": "value"}

        # Act
        data = component._convert_result_to_data(result)

        # Assert
        assert isinstance(data, Data)
        assert data.data == {"key": "value"}

    def test_should_wrap_list_in_results_key_when_result_is_list(self, component_class):
        # Arrange
        component = component_class()
        result = [1, 2, 3]

        # Act
        data = component._convert_result_to_data(result)

        # Assert
        assert isinstance(data, Data)
        assert data.data == {"_results": [1, 2, 3]}

    def test_should_convert_to_string_when_result_is_other_type(self, component_class):
        # Arrange
        component = component_class()
        result = 42

        # Act
        data = component._convert_result_to_data(result)

        # Assert
        assert isinstance(data, Data)
        assert data.data == {"text": "42"}


class TestConvertResultToDataframe(TestLambdaFilterComponent):
    """Tests for _convert_result_to_dataframe method."""

    def test_should_create_dataframe_when_result_is_list_of_dicts(self, component_class):
        # Arrange
        component = component_class()
        result = [{"a": 1}, {"a": 2}]

        # Act
        df = component._convert_result_to_dataframe(result)

        # Assert
        assert isinstance(df, DataFrame)

    def test_should_wrap_values_when_result_is_list_of_non_dicts(self, component_class):
        # Arrange
        component = component_class()
        result = [1, 2, 3]

        # Act
        df = component._convert_result_to_dataframe(result)

        # Assert
        assert isinstance(df, DataFrame)

    def test_should_create_single_row_when_result_is_dict(self, component_class):
        # Arrange
        component = component_class()
        result = {"a": 1}

        # Act
        df = component._convert_result_to_dataframe(result)

        # Assert
        assert isinstance(df, DataFrame)


class TestConvertResultToMessage(TestLambdaFilterComponent):
    """Tests for _convert_result_to_message method."""

    def test_should_return_message_when_result_is_string(self, component_class):
        # Arrange
        component = component_class()
        result = "Hello World"

        # Act
        msg = component._convert_result_to_message(result)

        # Assert
        assert isinstance(msg, Message)
        assert msg.text == "Hello World"

    def test_should_join_items_when_result_is_list(self, component_class):
        # Arrange
        component = component_class()
        result = ["Line 1", "Line 2"]

        # Act
        msg = component._convert_result_to_message(result)

        # Assert
        assert isinstance(msg, Message)
        assert msg.text == "Line 1\nLine 2"

    def test_should_format_json_when_result_is_dict(self, component_class):
        # Arrange
        component = component_class()
        result = {"key": "value"}

        # Act
        msg = component._convert_result_to_message(result)

        # Assert
        assert isinstance(msg, Message)
        assert '"key": "value"' in msg.text


class TestProcessAsDataIntegration(TestLambdaFilterComponent):
    """Integration tests for process_as_data method."""

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_return_filtered_data_when_lambda_is_valid(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: [item for item in x['items'] if item['value'] > 15]"

        # Act
        result = await component.process_as_data()

        # Assert
        assert isinstance(result, Data)
        assert "_results" in result.data
        filtered_items = result.data["_results"]
        assert len(filtered_items) == 1
        assert filtered_items[0]["name"] == "test2"
        assert filtered_items[0]["value"] == 20

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_raise_error_when_lambda_not_found_in_response(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "invalid response without lambda"

        # Act & Assert
        with pytest.raises(ValueError, match="Could not find lambda in response"):
            await component.process_as_data()


class TestProcessAsMessageIntegration(TestLambdaFilterComponent):
    """Integration tests for process_as_message with Message input."""

    @pytest.fixture
    def message_kwargs(self, model_metadata):
        """Return kwargs with Message input."""
        return {
            "data": [Message(text="Hello World")],
            "model": model_metadata,
            "api_key": "test-api-key",
            "filter_instruction": "Convert to uppercase",
            "sample_size": 1000,
            "max_size": 30000,
        }

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_transform_message_when_input_is_message(
        self, mock_get_model_class, component_class, message_kwargs, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        component = await self.component_setup(component_class, message_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda text: text.upper()"

        # Act
        result = await component.process_as_message()

        # Assert
        assert isinstance(result, Message)
        assert result.text == "HELLO WORLD"

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_join_multiple_messages_when_input_is_list_of_messages(
        self, mock_get_model_class, component_class, model_metadata, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        kwargs = {
            "data": [Message(text="Hello"), Message(text="World")],
            "model": model_metadata,
            "api_key": "test-api-key",
            "filter_instruction": "Convert to uppercase",
            "sample_size": 1000,
            "max_size": 30000,
        }
        component = await self.component_setup(component_class, kwargs)
        mock_llm.ainvoke.return_value.content = "lambda text: text.upper()"

        # Act
        result = await component.process_as_message()

        # Assert
        assert isinstance(result, Message)
        assert result.text == "HELLO\n\nWORLD"


class TestProcessAsDataframeIntegration(TestLambdaFilterComponent):
    """Integration tests for process_as_dataframe method."""

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_return_dataframe_when_lambda_returns_list_of_dicts(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: x['items']"

        # Act
        result = await component.process_as_dataframe()

        # Assert
        assert isinstance(result, DataFrame)


class TestLargeDataset(TestLambdaFilterComponent):
    """Tests for handling large datasets."""

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_filter_large_dataset_when_data_exceeds_max_size(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        large_data = {"items": [{"name": f"test{i}", "value": i} for i in range(2000)]}
        default_kwargs["data"] = [Data(data=large_data)]
        default_kwargs["filter_instruction"] = "Filter items with value greater than 1500"
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: [item for item in x['items'] if item['value'] > 1500]"

        # Act
        result = await component.process_as_data()

        # Assert
        assert isinstance(result, Data)
        filtered_items = result.data["_results"]
        assert len(filtered_items) == 499
        assert filtered_items[0]["value"] == 1501
        assert filtered_items[-1]["value"] == 1999


class TestComplexDataStructure(TestLambdaFilterComponent):
    """Tests for handling complex nested data structures."""

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_should_handle_nested_data_when_structure_is_complex(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        # Arrange
        mock_model_class = MagicMock(return_value=mock_llm)
        mock_get_model_class.return_value = mock_model_class
        complex_data = {
            "categories": {
                "A": [{"id": 1, "score": 90}, {"id": 2, "score": 85}],
                "B": [{"id": 3, "score": 95}, {"id": 4, "score": 88}],
            }
        }
        default_kwargs["data"] = [Data(data=complex_data)]
        default_kwargs["filter_instruction"] = "Filter items with score greater than 90"
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = (
            "lambda x: [item for cat in x['categories'].values() for item in cat if item['score'] > 90]"
        )

        # Act
        result = await component.process_as_data()

        # Assert
        assert isinstance(result, Data)
        filtered_items = result.data["_results"]
        assert len(filtered_items) == 1
        assert filtered_items[0]["id"] == 3
        assert filtered_items[0]["score"] == 95


def _run_guest_script(script: str) -> str:
    """Execute a generated guest script the way the sandbox guest would, and return its stdout.

    Running the real script, rather than asserting on its text, is what proves
    the data embedding survives hostile input instead of becoming code.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(script, "<guest>", "exec"), {})  # noqa: S102
    return buffer.getvalue()


def _guest_run(code, **_kwargs) -> SandboxResult:
    """Stand in for run_code_in_sandbox by running the script locally."""
    return SandboxResult(stdout=_run_guest_script(code), stderr="", exit_code=0)


class TestSandboxedLambdaExecution(TestLambdaFilterComponent):
    """The sandboxed path ships the lambda and its data into a VM instead of eval()ing in-process."""

    def test_lambda_is_applied_to_data_inside_the_guest(self, component_class):
        """Happy path: the generated script really transforms the data it carries."""
        component = component_class()

        with patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run):
            result = component._run_lambda_in_sandbox(
                "lambda x: [i for i in x['items'] if i['value'] > 15]",
                {"items": [{"name": "a", "value": 10}, {"name": "b", "value": 20}]},
            )

        assert result == [{"name": "b", "value": 20}]

    def test_message_text_round_trips_through_the_guest(self, component_class):
        """String input (the Message path) survives the JSON round trip."""
        component = component_class()

        with patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run):
            result = component._run_lambda_in_sandbox("lambda text: text.upper()", "hello world")

        assert result == "HELLO WORLD"

    @pytest.mark.parametrize(
        "hostile",
        [
            '"""',
            '") or __import__("os").system("id") or ("',
            "\\",
            "\\'\"",
            "line one\nline two",
            "'''\nimport os\nos.system('id')\n#",
            "\u2028\u2029",
            "\x00control",
        ],
    )
    def test_hostile_data_is_carried_as_data_not_code(self, component_class, hostile):
        """Data that looks like code must come back byte-identical, never execute."""
        component = component_class()

        with patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run):
            result = component._run_lambda_in_sandbox("lambda x: x['payload']", {"payload": hostile})

        assert result == hostile

    def test_incidental_guest_stdout_does_not_corrupt_the_result(self, component_class):
        """`print` is a valid expression in a lambda; its output must not be parsed as the payload."""
        component = component_class()

        with patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run):
            result = component._run_lambda_in_sandbox("lambda x: print('noise') or 'clean'", {"a": 1})

        assert result == "clean"

    def test_non_serializable_input_is_refused_before_reaching_the_guest(self, component_class):
        """A value the host cannot serialize fails with a clear message, not a mangled payload."""
        component = component_class()

        with (
            patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox") as mock_run,
            pytest.raises(ValueError, match="not JSON-serializable"),
        ):
            component._run_lambda_in_sandbox("lambda x: x", {"bad": {1, 2, 3}})

        mock_run.assert_not_called()

    def test_non_serializable_result_reports_a_clear_error(self, component_class):
        """A lambda returning a set fails inside the guest with an explanatory message."""
        component = component_class()

        def failing_run(code, **_kwargs):
            # Mirror the guest: the script raises, so the run exits non-zero with stderr.
            try:
                _run_guest_script(code)
            except TypeError as exc:
                return SandboxResult(stdout="", stderr=str(exc), exit_code=1)
            msg = "expected the guest script to raise"
            raise AssertionError(msg)

        with (
            patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=failing_run),
            pytest.raises(ValueError, match="not JSON-serializable"),
        ):
            component._run_lambda_in_sandbox("lambda x: set(x)", [1, 2, 3])

    def test_tuple_result_comes_back_as_a_list(self, component_class):
        """Documented behaviour difference from the in-process path: JSON has no tuple."""
        component = component_class()

        with patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run):
            result = component._run_lambda_in_sandbox("lambda x: tuple(x)", [1, 2])

        assert result == [1, 2]

    def test_guest_failure_surfaces_the_error_message(self, component_class):
        """A lambda that raises in the guest reports the guest's stderr."""
        component = component_class()
        failed = SandboxResult(stdout="", stderr="KeyError: 'missing'", exit_code=1)

        with (
            patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", return_value=failed),
            pytest.raises(ValueError, match="KeyError"),
        ):
            component._run_lambda_in_sandbox("lambda x: x['missing']", {})

    def test_non_json_guest_output_is_reported(self, component_class):
        """Truncated or corrupted guest stdout must not surface as a silent None."""
        component = component_class()
        garbled = SandboxResult(stdout="{not json", stderr="", exit_code=0)

        with (
            patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", return_value=garbled),
            pytest.raises(ValueError, match="not valid JSON"),
        ):
            component._run_lambda_in_sandbox("lambda x: x", {"a": 1})

    def test_unavailable_backend_fails_closed(self, component_class):
        """A configured-but-unusable backend propagates; it must never fall back to in-process eval."""
        component = component_class()

        with (
            patch(
                "lfx.components.llm_operations.lambda_filter.run_code_in_sandbox",
                side_effect=SandboxUnavailableError("backend down"),
            ),
            pytest.raises(SandboxUnavailableError),
        ):
            component._run_lambda_in_sandbox("lambda x: x", {"a": 1})

    def test_sandbox_error_is_not_disguised_as_a_conversion_failure(self, component_class):
        """`_handle_process_error` must re-raise infrastructure errors unchanged."""
        component = component_class()
        component.data = Data(data={"a": 1})

        with pytest.raises(SandboxUnavailableError, match="backend down"):
            component._handle_process_error(SandboxUnavailableError("backend down"), "Data")


class TestSandboxedProcessIntegration(TestLambdaFilterComponent):
    """End-to-end through process_as_* with the sandbox backend switched on."""

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_process_as_data_routes_through_the_sandbox(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        mock_get_model_class.return_value = MagicMock(return_value=mock_llm)
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: [i for i in x['items'] if i['value'] > 15]"

        with (
            patch("lfx.components.llm_operations.lambda_filter.is_sandbox_enabled", return_value=True),
            patch(
                "lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run
            ) as mock_run,
        ):
            result = await component.process_as_data()

        mock_run.assert_called_once()
        assert isinstance(result, Data)
        assert result.data["_results"] == [{"name": "test2", "value": 20}]

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_process_as_message_routes_through_the_sandbox(
        self, mock_get_model_class, component_class, model_metadata, mock_llm
    ):
        mock_get_model_class.return_value = MagicMock(return_value=mock_llm)
        kwargs = {
            "data": [Message(text="Hello World")],
            "model": model_metadata,
            "api_key": "test-api-key",
            "filter_instruction": "Convert to uppercase",
            "sample_size": 1000,
            "max_size": 30000,
        }
        component = await self.component_setup(component_class, kwargs)
        mock_llm.ainvoke.return_value.content = "lambda text: text.upper()"

        with (
            patch("lfx.components.llm_operations.lambda_filter.is_sandbox_enabled", return_value=True),
            patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox", side_effect=_guest_run),
        ):
            result = await component.process_as_message()

        assert isinstance(result, Message)
        assert result.text == "HELLO WORLD"

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_unavailable_backend_propagates_through_process_as_data(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        """Fail closed end to end: no fallback, and no misleading conversion error."""
        mock_get_model_class.return_value = MagicMock(return_value=mock_llm)
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: x"

        with (
            patch("lfx.components.llm_operations.lambda_filter.is_sandbox_enabled", return_value=True),
            patch(
                "lfx.components.llm_operations.lambda_filter.run_code_in_sandbox",
                side_effect=SandboxUnavailableError("backend down"),
            ),
            pytest.raises(SandboxUnavailableError, match="backend down"),
        ):
            await component.process_as_data()

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_in_process_path_is_unchanged_when_sandbox_is_off(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        """With the backend off the component still eval()s in-process and never calls the sandbox."""
        mock_get_model_class.return_value = MagicMock(return_value=mock_llm)
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: [i for i in x['items'] if i['value'] > 15]"

        with (
            patch("lfx.components.llm_operations.lambda_filter.is_sandbox_enabled", return_value=False),
            patch("lfx.components.llm_operations.lambda_filter.run_code_in_sandbox") as mock_run,
        ):
            result = await component.process_as_data()

        mock_run.assert_not_called()
        assert result.data["_results"] == [{"name": "test2", "value": 20}]

    @patch("lfx.base.models.unified_models.get_model_class")
    async def test_escape_gadget_reaches_the_guest_instead_of_being_rejected(
        self, mock_get_model_class, component_class, default_kwargs, mock_llm
    ):
        """The AST check is a host protection; under the VM it is intentionally not applied."""
        mock_get_model_class.return_value = MagicMock(return_value=mock_llm)
        component = await self.component_setup(component_class, default_kwargs)
        mock_llm.ainvoke.return_value.content = "lambda x: x.__class__.__name__"

        with (
            patch("lfx.components.llm_operations.lambda_filter.is_sandbox_enabled", return_value=True),
            patch(
                "lfx.components.llm_operations.lambda_filter.run_code_in_sandbox",
                return_value=SandboxResult(stdout='"dict"', stderr="", exit_code=0),
            ) as mock_run,
        ):
            result = await component.process_as_data()

        mock_run.assert_called_once()
        assert result.data == {"text": "dict"}
