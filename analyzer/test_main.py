import pytest
from httpx import AsyncClient, ASGITransport
from main import app  # Import your FastAPI app

@pytest.mark.asyncio
async def test_predict_text():
    """
    Test the POST request to the /predict endpoint.
    """
    # Use ASGITransport to connect httpx to the FastAPI app explicitly
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/predict", json={"text": "I love playing hockey!"})
    assert response.status_code == 200
    assert "prediction" in response.json()  
    assert response.json()["prediction"] == "rec.sport.hockey" 
