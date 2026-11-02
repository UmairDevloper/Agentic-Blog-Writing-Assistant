import base64
from io import BytesIO

from PIL import Image

from scripts import check_image_provider
from seo_blog_assistant.schemas import GeneratedImage


def _generated_image() -> GeneratedImage:
    buffer = BytesIO()
    Image.new("RGB", (2, 3), color="red").save(buffer, format="PNG")
    return GeneratedImage(
        model_name="owner/test-model",
        prompt="A red apple on a wooden table",
        alt_text="A red apple",
        image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def test_image_provider_check_saves_validated_png_without_network(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class FakeImageProvider:
        async def generate_image(
            self,
            _prompt: str,
            _alt_text: str,
        ) -> GeneratedImage:
            return _generated_image()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_image_provider, "Settings", lambda: object())
    monkeypatch.setattr(
        check_image_provider,
        "create_image_provider",
        lambda _settings: FakeImageProvider(),
    )

    assert check_image_provider.main() == 0
    assert capsys.readouterr().out.strip() == "OK 2x3"
    with Image.open(tmp_path / "test_image.png") as image:
        assert image.size == (2, 3)


def test_image_provider_check_refuses_to_overwrite_existing_file(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    output_path = tmp_path / "test_image.png"
    output_path.write_bytes(b"keep this file")
    provider_factory_called = False

    def create_provider(_settings: object) -> object:
        nonlocal provider_factory_called
        provider_factory_called = True
        raise AssertionError("existing file should be detected first")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_image_provider, "Settings", lambda: object())
    monkeypatch.setattr(
        check_image_provider,
        "create_image_provider",
        create_provider,
    )

    assert check_image_provider.main() == 1
    assert "already exists" in capsys.readouterr().out
    assert not provider_factory_called
    assert output_path.read_bytes() == b"keep this file"
