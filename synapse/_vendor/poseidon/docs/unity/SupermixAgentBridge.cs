using System;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Supermix Beyond - Unity WorldLab Agent Bridge.
/// 
/// Attach to your Agent GameObject in Unity.
/// Connects via local TCP socket to Supermix Beyond's Python backend (127.0.0.1:8085).
/// Transmits 16-d normalized sensor observations and receives uncertainty-calibrated macro actions.
/// </summary>
public class SupermixAgentBridge : MonoBehaviour
{
    [Header("Network Configuration")]
    public string host = "127.0.0.1";
    public int port = 8085;
    public float decisionInterval = 0.2f;

    [Header("Agent State Monitoring")]
    [Range(0f, 1f)] public float health = 1.0f;
    [Range(0f, 1f)] public float energy = 0.8f;
    [Range(0f, 1f)] public float hydration = 0.8f;
    [Range(0f, 1f)] public float stamina = 0.9f;
    [Range(0f, 1f)] public float exposure = 0.1f;
    [Range(0f, 1f)] public float localThreat = 0.0f;

    [Header("Live Beyond Telemetry")]
    public string currentAction = "rest";
    public float epistemicUncertainty = 0.0f;
    public float failureRisk = 0.0f;
    public bool isHighSurprise = false;

    private TcpClient client;
    private NetworkStream stream;
    private StreamWriter writer;
    private StreamReader reader;
    private float timer = 0f;

    private void Start()
    {
        ConnectToBeyond();
    }

    private void ConnectToBeyond()
    {
        try
        {
            client = new TcpClient(host, port);
            stream = client.GetStream();
            writer = new StreamWriter(stream, Encoding.UTF8) { AutoFlush = true };
            reader = new StreamReader(stream, Encoding.UTF8);

            // Send Handshake
            writer.WriteLine("{\"command\":\"handshake\"}");
            string response = reader.ReadLine();
            Debug.Log($"[Supermix Beyond Bridge] Connected: {response}");
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[Supermix Beyond Bridge] Connection failed: {ex.Message}. Retrying...");
        }
    }

    private void Update()
    {
        if (client == null || !client.Connected)
        {
            return;
        }

        timer += Time.deltaTime;
        if (timer >= decisionInterval)
        {
            timer = 0f;
            RequestAction();
        }
    }

    private void RequestAction()
    {
        try
        {
            // Pack 16-element normalized observation
            float[] obs = new float[16] {
                health, energy, hydration, stamina, exposure, localThreat,
                0.5f, 0.5f, 0.8f, // food, water, shelter in area
                0.1f, 0.5f, 0.5f, // weather severity, temp, daylight
                0.2f, 0.4f,       // terrain difficulty, scent
                0.0f, Time.time / 100f // last action scaled, progress
            };

            string jsonObs = "[" + string.Join(",", obs) + "]";
            Vector3 pos = transform.position;
            string payload = $"{{\"command\":\"step\",\"observation\":{jsonObs},\"position\":[{pos.x},{pos.z}],\"reward\":0.0}}";

            writer.WriteLine(payload);
            string responseJson = reader.ReadLine();

            if (!string.IsNullOrEmpty(responseJson))
            {
                StepResponse res = JsonUtility.FromJson<StepResponse>(responseJson);
                currentAction = res.action_name;
                epistemicUncertainty = res.epistemic_uncertainty;
                failureRisk = res.failure_probability;
                isHighSurprise = res.is_high_surprise;

                ExecuteAction(res.action);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"[Supermix Beyond Bridge] Error querying planner: {ex.Message}");
        }
    }

    private void ExecuteAction(int actionIndex)
    {
        // 0: rest, 1: forage, 2: drink, 3: shelter, 4: explore, 5: flee
        switch (actionIndex)
        {
            case 0: // Rest
                break;
            case 1: // Forage
                energy = Mathf.Min(1.0f, energy + 0.15f);
                break;
            case 2: // Drink
                hydration = Mathf.Min(1.0f, hydration + 0.20f);
                break;
            case 3: // Shelter
                exposure = Mathf.Max(0.0f, exposure - 0.25f);
                break;
            case 4: // Explore
                transform.Translate(transform.forward * 1.5f * Time.deltaTime, Space.World);
                stamina = Mathf.Max(0.0f, stamina - 0.05f);
                break;
            case 5: // Flee
                transform.Translate(-transform.forward * 2.5f * Time.deltaTime, Space.World);
                stamina = Mathf.Max(0.0f, stamina - 0.10f);
                break;
        }
    }

    private void OnDestroy()
    {
        try
        {
            writer?.Close();
            reader?.Close();
            client?.Close();
        }
        catch { }
    }

    [System.Serializable]
    private class StepResponse
    {
        public string status;
        public int action;
        public string action_name;
        public float epistemic_uncertainty;
        public float failure_probability;
        public bool is_high_surprise;
    }
}
