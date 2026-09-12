//! A coding agent's control handle is a session on the steering bus.
//!
//! The bus steers whatever implements its contract; pebble's own handle does
//! natively, so an application attaches the handle and the bus's steers,
//! interrupts, and holds are the handle's own.

use std::sync::Arc;

use pebble_coding_agent::steering::{SteerableSession, SteeringBus};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, client_from, text_response,
};
use pebble_coding_agent::{CodingAgent, ShutdownReason, SteeringMessage};

#[tokio::test]
async fn the_control_handle_is_steered_and_held_through_the_bus() {
    let (client, _provider) = client_from(ScriptedProvider::new(vec![ScriptedCall::response(
        text_response("done"),
    )]));
    let mut agent = CodingAgent::builder(client, Arc::new(MockEnvironment::linux()))
        .model("test/model")
        .build()
        .await
        .expect("the coding agent builds");
    let handle = agent.control_handle();
    let bus: SteeringBus<&str> = SteeringBus::new();

    bus.steer(SteeringMessage::new("before anyone was listening"));
    bus.attach("stage", agent.id(), Arc::new(handle.clone()))
        .expect("attaches");
    let drained = bus.drain_pending_into(&"stage");
    assert_eq!(drained.delivered.len(), 1);
    assert!(
        handle.has_pending_steering(),
        "the buffered steer is queued on the agent"
    );
    assert_eq!(handle.snapshot().pending_steering(), 1);
    assert!(!bus.detach_if_idle(&"stage", agent.id()), "steering waits");

    bus.hold_open(&"stage", agent.id())
        .expect("a coding agent can be held");
    assert!(bus.is_held(&"stage", agent.id()));
    assert!(bus.release_hold(&"stage", agent.id()));

    assert!(
        bus.interrupt().interrupted.len() == 1,
        "an idle agent is still told; there is no round to stop"
    );

    let taken = handle.take_pending_input();
    assert_eq!(taken.steering().len(), 1);
    assert!(bus.detach_if_idle(&"stage", agent.id()));

    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}
